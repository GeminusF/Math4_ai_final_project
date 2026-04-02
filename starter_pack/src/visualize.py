"""
Plotting utilities for all our experiments.

We define a custom style at the top so everything looks consistent
and professional across the report. Went with a clean academic look
with a subtle grid on a light background.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

# ── our custom style ───────────────────────────────────────────────
# aiming for a polished-but-not-overdone academic look
PALETTE = ["#2563eb", "#f97316", "#10b981", "#ef4444", "#8b5cf6", "#ec4899"]
BG_COLOR = "#fafafa"
GRID_COLOR = "#e2e2e2"
TEXT_COLOR = "#1e293b"
ACCENT = "#2563eb"

def _apply_style():
    """Set matplotlib rcParams for our project-wide look."""
    rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": "#cbd5e1",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.labelcolor": TEXT_COLOR,
        "axes.titlecolor": TEXT_COLOR,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#cbd5e1",
        "legend.fontsize": 9,
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
        "savefig.facecolor": BG_COLOR,
    })

_apply_style()


def _save(fig, filename):
    if filename:
        FIG_DIR.mkdir(exist_ok=True)
        fig.savefig(FIG_DIR / filename)
        print(f"    Saved: {filename}")


# ── decision boundary ──────────────────────────────────────────────
def plot_decision_boundary(model, X, y, title="Decision Boundary",
                           filename=None, ax=None):
    """2D decision boundary with data scatter."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 5.5))

    pad = 0.5
    x_lo, x_hi = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_lo, y_hi = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_lo, x_hi, 300),
                         np.linspace(y_lo, y_hi, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    # filled contour for regions
    cmap_bg = matplotlib.colors.ListedColormap(["#dbeafe", "#fee2e2"])
    ax.contourf(xx, yy, Z, alpha=0.45, cmap=cmap_bg, levels=[-0.5, 0.5, 1.5])
    ax.contour(xx, yy, Z, colors="#94a3b8", linewidths=1.2, levels=[0.5])

    # scatter with class coloring
    colors_scatter = [PALETTE[0], PALETTE[3]]
    for cls_val in np.unique(y):
        mask = y == cls_val
        ax.scatter(X[mask, 0], X[mask, 1], c=colors_scatter[int(cls_val)],
                   edgecolors="#334155", s=22, linewidths=0.5,
                   label=f"Class {cls_val}", alpha=0.85, zorder=3)

    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc="best", fontsize=8)

    if own_fig:
        fig.tight_layout()
        _save(fig, filename)
        plt.close(fig)


# ── training curves ────────────────────────────────────────────────
def plot_training_curves(history, title="Training Curves", filename=None):
    """Side-by-side loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    eps = range(1, len(history["train_loss"]) + 1)

    ax1.plot(eps, history["train_loss"], linewidth=1.8, label="Train")
    ax1.plot(eps, history["val_loss"], linewidth=1.8, label="Validation",
             linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title(f"{title} — Loss")
    ax1.legend()

    ax2.plot(eps, history["train_acc"], linewidth=1.8, label="Train")
    ax2.plot(eps, history["val_acc"], linewidth=1.8, label="Validation",
             linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{title} — Accuracy")
    ax2.legend()

    fig.tight_layout(w_pad=3)
    _save(fig, filename)
    plt.close(fig)


# ── optimizer comparison ───────────────────────────────────────────
def plot_optimizer_comparison(histories, opt_names,
                              title="Optimizer Comparison", filename=None):
    """Overlay val loss and accuracy for multiple optimizers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for hist, name in zip(histories, opt_names):
        eps = range(1, len(hist["val_loss"]) + 1)
        ax1.plot(eps, hist["val_loss"], linewidth=1.8, label=name)
        ax2.plot(eps, hist["val_acc"], linewidth=1.8, label=name)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Cross-Entropy")
    ax1.set_title(f"{title} — Val Loss")
    ax1.legend()

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy")
    ax2.set_title(f"{title} — Val Accuracy")
    ax2.legend()

    fig.tight_layout(w_pad=3)
    _save(fig, filename)
    plt.close(fig)


# ── capacity ablation ──────────────────────────────────────────────
def plot_capacity_ablation(models, widths, X, y, filename=None):
    """Side-by-side decision boundaries for different hidden widths."""
    n = len(widths)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5))
    if n == 1:
        axes = [axes]

    for ax, mdl, w in zip(axes, models, widths):
        plot_decision_boundary(mdl, X, y, title=f"Hidden Width = {w}", ax=ax)

    fig.tight_layout(w_pad=2)
    _save(fig, filename)
    plt.close(fig)


# ── reliability diagram ───────────────────────────────────────────
def plot_confidence_reliability(confidences, correctness, model_name,
                                n_bins=5, filename=None):
    """5-bin reliability diagram: confidence vs empirical accuracy."""
    edges = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)

        cnt = int(mask.sum())
        bin_counts.append(cnt)
        if cnt > 0:
            bin_accs.append(float(correctness[mask].mean()))
            bin_confs.append(float(confidences[mask].mean()))
        else:
            bin_accs.append(0.0)
            bin_confs.append((lo + hi) / 2)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    centers = [(edges[i] + edges[i+1]) / 2 for i in range(n_bins)]
    bar_w = 0.75 / n_bins

    # gap bars
    gap_colors = []
    for acc, conf in zip(bin_accs, bin_confs):
        gap_colors.append("#ef4444" if acc < conf else ACCENT)

    bars = ax.bar(centers, bin_accs, width=bar_w, alpha=0.8,
                  color=ACCENT, edgecolor="#334155", linewidth=0.8,
                  label="Empirical accuracy", zorder=3)

    # color bars by over/under confidence
    for bar, gc in zip(bars, gap_colors):
        bar.set_facecolor(gc)

    ax.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", linewidth=1.5,
            label="Perfect calibration", zorder=2)

    # add count annotations
    for c, cnt, acc in zip(centers, bin_counts, bin_accs):
        if cnt > 0:
            ax.text(c, acc + 0.03, f"n={cnt}", ha="center", va="bottom",
                    fontsize=8, color=TEXT_COLOR)

    ax.set_xlabel("Confidence (max predicted probability)")
    ax.set_ylabel("Empirical Accuracy")
    ax.set_title(f"Reliability Diagram — {model_name}")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)

    fig.tight_layout()
    _save(fig, filename)
    plt.close(fig)

    return {
        "bin_edges": edges,
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_counts": bin_counts,
    }