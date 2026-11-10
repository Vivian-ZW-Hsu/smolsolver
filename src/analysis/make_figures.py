#!/usr/bin/env python3
"""
Make the README figures
=======================
Reads only result files in this repository and writes three PNGs to figures/:

  figures/pass_at_k.png            Pass@k on the test set vs. majority-vote accuracy
  figures/selection_methods.png    accuracy of the 7 selection methods, overall and by group
  figures/error_concentration.png  where majority-vote errors fall by consensus group

Inputs:
  results/test_sample_stats.json   (from src/analysis/summarize_test_samples.py)
  results/ranker_eval_results.json (from src/evaluation/evaluate_selectors.py)
  results/pass_at_k.json           (original README table; used as a cross-check)

Usage (from anywhere):
    python src/analysis/summarize_test_samples.py   # once, if test_sample_stats.json is missing
    python src/analysis/make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DPI = 200

# Colors. Group colors are one blue ramp (light = high agreement, dark = low
# agreement); "overall" uses orange so it never reads as a fourth group.
BLUE = "#4C78A8"
ORANGE = "#F58518"
GROUP_COLORS = {"high": "#86B6EF", "medium": "#4C78A8", "low": "#104281"}
GROUP_TEXT = {"high": "#1A1A1A", "medium": "#FFFFFF", "low": "#FFFFFF"}  # label ink on each fill
INK = "#1A1A1A"
INK_SECONDARY = "#555555"
GRID = "#E3E3E3"
REFERENCE = "#6B6B6B"

GROUP_LABELS = {"high": "High agreement", "medium": "Medium agreement", "low": "Low agreement"}

METHODS = [
    ("mv", "Majority\nvote"),
    ("cls_top", "Classifier\ntop-1"),
    ("cls_wmv", "Classifier-\nweighted vote"),
    ("rnk_top", "Ranker\ntop-1"),
    ("rnk_wmv", "Ranker-\nweighted vote"),
    ("cls_rnk_top", "Classifier filter,\nranker top-1"),
    ("cls_rnk_wmv", "Classifier filter,\nranker-weighted vote"),
]


def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.edgecolor": "#BDBDBD",
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.titlelocation": "left",
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "legend.frameon": False,
    })


def clean_axes(ax, ygrid=True):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    ax.xaxis.grid(False)


def add_titles(fig, title, subtitle, x=0.06, top=0.965):
    fig.text(x, top, title, fontsize=15, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(x, top - 0.065, subtitle, fontsize=10.5, color=INK_SECONDARY, ha="left", va="top")


# ---------------------------------------------------------------------------
# Figure 1: Pass@k vs. majority vote
# ---------------------------------------------------------------------------
def fig_pass_at_k(stats, evals, readme_table):
    ks = [1, 3, 5, 7, 10, 15]
    test = stats["pass_at_k_percent"]["overall"]
    counts = stats["pass_at_k_count"]["overall"]
    n = stats["n_problems"]["overall"]
    ys = [100 * counts[str(k)] / n for k in ks]  # unrounded, so labels round correctly

    # The original README reported test Pass@1/10/15; make sure we agree.
    for k, v in readme_table["test_1319q"].items():
        if k.isdigit():
            assert abs(test[k] - v) < 0.006, f"Pass@{k} mismatch: {test[k]} vs README {v}"

    mv = 100 * evals["accuracy"]["mv"]["overall"]
    learned = {m: 100 * evals["accuracy"][m]["overall"] for m, _ in METHODS if m != "mv"}
    best_learned = max(learned.values())

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.8, bottom=0.13)

    ax.axhline(mv, color=REFERENCE, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.text(3.8, mv - 1.8,
            f"Majority vote over 15 samples: {mv:.1f}%\n(best learned selector: {best_learned:.1f}%)",
            fontsize=10, color=INK_SECONDARY, ha="left", va="top", linespacing=1.35)

    ax.plot(ks, ys, color=BLUE, linewidth=2.2, solid_capstyle="round", solid_joinstyle="round", zorder=3)
    ax.scatter(ks, ys, s=64, color=BLUE, edgecolor="white", linewidth=2, zorder=4)

    ax.annotate(f"Pass@1  {ys[0]:.1f}%", (ks[0], ys[0]), xytext=(10, -4), textcoords="offset points",
                fontsize=10.5, color=INK, ha="left", va="top")
    ax.annotate(f"Pass@15  {ys[-1]:.1f}%", (ks[-1], ys[-1]), xytext=(0, 10), textcoords="offset points",
                fontsize=10.5, color=INK, ha="right", va="bottom")

    # Gap between "a correct answer was sampled" and "majority vote picked it".
    gx = 15.55
    ax.annotate("", xy=(gx, mv + 0.3), xytext=(gx, ys[-1] - 0.3),
                arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.1, shrinkA=0, shrinkB=0))
    ax.text(gx - 0.35, (mv + ys[-1]) / 2,
            f"{ys[-1] - mv:.1f}-pt gap:\ncorrect answer sampled\nbut not selected",
            fontsize=10, color=INK, ha="right", va="center", linespacing=1.3)

    ax.set_xlim(0.4, 16.2)
    ax.set_ylim(0, 100)
    ax.set_xticks(ks)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel("Samples per problem (k)")
    ax.set_ylabel("Test problems with a correct sample (%)")
    clean_axes(ax)

    add_titles(fig, "The right answer is usually sampled, but often not selected",
               "Pass@k of the fine-tuned Phi-2 generator vs. majority-vote accuracy, "
               "GSM8K test (1,319 problems, 15 samples each)")
    out = FIGURES / "pass_at_k.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2: the 7 selection methods, overall and by consensus group
# ---------------------------------------------------------------------------
def fig_selection_methods(evals):
    acc = evals["accuracy"]
    sizes = evals["group_sizes"]
    n_total = sum(sizes.values())
    series = [("overall", f"Overall (n = {n_total:,})", ORANGE)] + [
        (g, f"{GROUP_LABELS[g]} (n = {sizes[g]})", GROUP_COLORS[g]) for g in ("high", "medium", "low")
    ]

    fig, ax = plt.subplots(figsize=(12, 5.6))
    fig.subplots_adjust(left=0.065, right=0.99, top=0.79, bottom=0.12)

    n_series = len(series)
    width = 0.19
    xs = range(len(METHODS))
    for j, (key, label, color) in enumerate(series):
        offs = (j - (n_series - 1) / 2) * width
        vals = [100 * acc[m][key] for m, _ in METHODS]
        bars = ax.bar([x + offs for x in xs], vals, width=width, color=color, label=label,
                      edgecolor="white", linewidth=1.5, zorder=3)
        if key == "overall":
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", fontsize=9.5,
                        color=INK, ha="center", va="bottom", fontweight="bold")

    ax.set_xticks(list(xs))
    ax.set_xticklabels([lbl for _, lbl in METHODS], fontsize=10, color=INK)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Accuracy (%)")
    clean_axes(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=4, fontsize=10, handlelength=1.2,
              handleheight=1.0, columnspacing=1.6, borderaxespad=0)

    add_titles(fig, "No learned selector beats plain majority voting",
               "Final-answer accuracy of 7 answer-selection methods over the same 15 samples per problem "
               "(GSM8K test). Overall values labeled.",
               x=0.065)
    out = FIGURES / "selection_methods.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 3: where majority-vote errors fall
# ---------------------------------------------------------------------------
def fig_error_concentration(stats):
    groups = ("high", "medium", "low")
    n = stats["n_problems"]
    mv = stats["majority_vote"]
    rows = [
        (f"All test problems\n(n = {n['overall']:,})", {g: n[g] for g in groups}),
        (f"Majority-vote errors\n(n = {mv['overall']['errors']})", {g: mv[g]["errors"] for g in groups}),
    ]

    fig, ax = plt.subplots(figsize=(10, 3.9))
    fig.subplots_adjust(left=0.19, right=0.97, top=0.71, bottom=0.17)

    bar_h = 0.56
    for i, (row_label, counts) in enumerate(rows):
        y = len(rows) - 1 - i
        total = sum(counts.values())
        left = 0.0
        for g in groups:
            share = 100 * counts[g] / total
            ax.barh(y, share, left=left, height=bar_h, color=GROUP_COLORS[g], edgecolor="white",
                    linewidth=2, zorder=3, label=GROUP_LABELS[g] if i == 0 else None)
            text = f"{share:.1f}%  ({counts[g]})"
            if share >= 12:
                ax.text(left + share / 2, y, text, ha="center", va="center", fontsize=10.5,
                        color=GROUP_TEXT[g], fontweight="bold", zorder=4)
            else:  # too narrow for an inside label: put it just above the segment
                ax.annotate(text, (left + share / 2, y + bar_h / 2), xytext=(0, 4),
                            textcoords="offset points", ha="center", va="bottom", fontsize=10,
                            color=INK)
            left += share

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=10.5, color=INK)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, len(rows) - 0.5 + 0.12)
    ax.set_xticks(range(0, 101, 25))
    ax.set_xlabel("Share of row total (%)")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, fontsize=10, handlelength=1.2,
              handleheight=1.0, columnspacing=1.6, borderaxespad=0)

    low_share = mv["low"]["share_of_all_errors_percent"]
    add_titles(fig, f"{low_share:.0f}% of majority-vote errors come from low-agreement problems",
               f"Low-agreement problems are {mv['low']['share_of_problems_percent']:.0f}% of the test set "
               f"but {mv['low']['errors']} of {mv['overall']['errors']} majority-vote errors. "
               "Groups: 33rd/66th percentiles of the top-2 answer margin.",
               x=0.03, top=0.955)
    out = FIGURES / "error_concentration.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def main():
    style()
    FIGURES.mkdir(exist_ok=True)
    stats = load("test_sample_stats.json")
    evals = load("ranker_eval_results.json")
    readme_table = load("pass_at_k.json")
    for out in (fig_pass_at_k(stats, evals, readme_table),
                fig_selection_methods(evals),
                fig_error_concentration(stats)):
        print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
