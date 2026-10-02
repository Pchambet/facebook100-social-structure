"""Static figures for the README. Titles state the finding and are filled from the results."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fb100.pipeline import load_results

INK, TEAL, AMBER, SLATE, GRID = "#0f172a", "#0d9488", "#d97706", "#64748b", "#e2e8f0"
LIGHT_SLATE = "#cbd5e1"

ATTR_ORDER = ["year", "status", "dorm", "degree", "gender", "major", "minor", "high_school"]
ATTR_LABEL = {
    "year": "Class year",
    "status": "Status (student, faculty, ...)",
    "dorm": "Dorm / house",
    "degree": "Degree (popularity)",
    "gender": "Gender",
    "major": "Major",
    "minor": "Second major / minor",
    "high_school": "High school",
}
METHOD_LABEL = {
    "resource_allocation": "Resource allocation",
    "adamic_adar": "Adamic-Adar",
    "jaccard": "Jaccard",
    "common_neighbors": "Common neighbours",
    "preferential_attachment": "Preferential attachment",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": SLATE,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "font.size": 10.5,
            "font.family": "DejaVu Sans",
            "legend.frameon": False,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def hero_assortativity(res: dict[str, pd.DataFrame], s: dict, path: Path) -> None:
    asr = res["assortativity"]
    asr = asr[asr["attribute"].isin(ATTR_ORDER)].dropna(subset=["r"])
    fig, ax = plt.subplots(figsize=(10, 5.2))
    rng = np.random.default_rng(0)
    highlight = {"year", "dorm"}
    for i, attr in enumerate(ATTR_ORDER):
        y = len(ATTR_ORDER) - 1 - i
        r = asr.loc[asr["attribute"] == attr, "r"].to_numpy()
        color = TEAL if attr in highlight else SLATE
        ax.scatter(
            r,
            y + rng.uniform(-0.18, 0.18, len(r)),
            s=14,
            color=color,
            alpha=0.45 if attr in highlight else 0.35,
            linewidths=0,
        )
        med = np.median(r)
        ax.plot([med, med], [y - 0.32, y + 0.32], color=INK, lw=2.2, solid_capstyle="round")
        ax.annotate(
            f"{med:.2f}",
            (med, y + 0.34),
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK,
        )
    ax.axvline(0, color=SLATE, lw=1, ls="--")
    ax.set_yticks(range(len(ATTR_ORDER)), [ATTR_LABEL[a] for a in ATTR_ORDER[::-1]])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Assortativity coefficient r  (0 = random mixing, 1 = only within group)")
    med = asr.groupby("attribute")["r"].median()
    year, dorm, major = med["year"], med["dorm"], med["major"]
    ax.set_title(
        f"Across {s['schools']} campuses, friends share a class year (median r = {year:.2f}) "
        f"and a dorm ({dorm:.2f}),\nfar more than a major ({major:.2f})",
        pad=12,
    )
    ax.text(
        1.0,
        -0.14,
        "One dot per campus; black bar = median. Missing values excluded. Facebook100, Sept. 2005.",
        transform=ax.transAxes,
        ha="right",
        fontsize=8.5,
        color=SLATE,
    )
    _save(fig, path)


def erratum(res: dict[str, pd.DataFrame], s: dict, path: Path) -> None:
    a = s["assortativity"]
    # (label, first version, column order fixed, missing values also excluded)
    rows = [
        ("Class year", "year (legacy label)", "year (missing counted)", "year"),
        ("Dorm / house", "dorm (legacy label)", "dorm (missing counted)", "dorm"),
    ]
    steps = [
        ("first version", SLATE, "normal", 1),
        ("column order fixed", AMBER, "normal", -1),
        ("missing excluded", TEAL, "bold", 1),
    ]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for y, (_label, *keys) in enumerate(rows[::-1]):
        values = [a[k]["mean"] for k in keys]
        for x0, x1 in itertools.pairwise(values):
            ax.annotate(
                "",
                xy=(x1, y),
                xytext=(x0, y),
                arrowprops={
                    "arrowstyle": "->",
                    "color": LIGHT_SLATE,
                    "lw": 2,
                    "shrinkA": 6,
                    "shrinkB": 6,
                },
            )
        for v, (name, color, weight, side) in zip(values, steps, strict=True):
            ax.scatter([v], [y], s=80, color=color, zorder=2)
            ax.annotate(
                f"{name}\n{v:.3f}" if side > 0 else f"{v:.3f}\n{name}",
                (v, y + 0.14 * side),
                ha="center",
                va="bottom" if side > 0 else "top",
                fontsize=9,
                color=INK if weight == "bold" else color,
                fontweight=weight,
            )
    ax.set_yticks([0, 1], [r[0] for r in rows[::-1]])
    ax.set_ylim(-0.75, 1.75)
    ax.set_xlim(-0.03, max(a["year"]["mean"], a["dorm"]["mean"]) * 1.15)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Mean assortativity r over {s['schools']} campuses")
    ax.set_title(
        "Erratum: the first version read the columns one slot off and counted missing\n"
        "values as a category, which understated the class-year and dorm effects",
        pad=10,
    )
    _save(fig, path)


def link_prediction(res: dict[str, pd.DataFrame], s: dict, path: Path) -> None:
    full = res["linkpred_full"]
    mean = full.groupby("method")[["auc_sampled", "r_precision", "precision_at_100"]].mean()
    sd = full.groupby(["method", "school"])[["r_precision"]].mean().groupby("method").std()
    order = mean.sort_values("r_precision").index.tolist()
    base = full["base_rate"].mean()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    y = np.arange(len(order))
    labels = [METHOD_LABEL[m] for m in order]
    ax1.barh(y, mean.loc[order, "auc_sampled"], color=LIGHT_SLATE, height=0.55)
    for yi, v in zip(y, mean.loc[order, "auc_sampled"], strict=True):
        ax1.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=9)
    ax1.set_xlim(0.5, 1.05)
    ax1.set_yticks(y, labels)
    ax1.set_xlabel("ROC AUC, hidden edges vs random non-edges")
    ax1.set_title("Easy benchmark: sampled AUC", fontsize=11.5)
    colors = [TEAL if m != "preferential_attachment" else SLATE for m in order]
    ax2.barh(
        y,
        mean.loc[order, "r_precision"],
        color=colors,
        height=0.55,
        xerr=sd.loc[order, "r_precision"],
        error_kw={"ecolor": INK, "lw": 1},
    )
    for yi, v, e in zip(
        y, mean.loc[order, "r_precision"], sd.loc[order, "r_precision"], strict=True
    ):
        ax2.text(v + e + 0.006, yi, f"{v:.3f}", va="center", fontsize=9)
    ax2.axvline(base, color=AMBER, lw=1.5, ls="--")
    ax2.annotate(
        f"random ranking: {base:.4f}",
        (base, -0.45),
        xytext=(6, 0),
        textcoords="offset points",
        color=AMBER,
        fontsize=9,
    )
    ax2.set_xlabel("R-precision on the full ranking of unlinked pairs")
    ax2.set_title("Real task: rank every unlinked pair", fontsize=11.5)
    for ax in (ax1, ax2):
        ax.grid(axis="y", visible=False)
    best = mean["r_precision"].max()
    fig.suptitle(
        f"Neighbourhood scores reach AUC ~{mean['auc_sampled'].max():.2f}, yet only "
        f"{best:.0%} of their top-ranked pairs are hidden friendships "
        f"({s['linkpred_best_lift_over_random']}x random)",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color=INK,
        y=1.04,
    )
    fig.text(
        0.01,
        -0.06,
        "10 smallest campuses x 5 random 10% edge splits; error bars = s.d. across campuses.",
        fontsize=8.5,
        color=SLATE,
    )
    _save(fig, path)


def label_propagation(res: dict[str, pd.DataFrame], s: dict, path: Path) -> None:
    lp = res["labelprop"]
    attrs = ["year", "dorm", "gender", "major"]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    rng = np.random.default_rng(1)
    for i, attr in enumerate(attrs):
        d = lp[lp["attribute"] == attr]
        y = len(attrs) - 1 - i
        jitter = rng.uniform(-0.15, 0.15, len(d))
        ax.scatter(d["majority_baseline"], y + jitter, s=10, color=SLATE, alpha=0.35, linewidths=0)
        ax.scatter(d["accuracy"], y + jitter, s=10, color=TEAL, alpha=0.45, linewidths=0)
        mb, acc = d["majority_baseline"].median(), d["accuracy"].median()
        ax.annotate(
            "",
            xy=(acc, y),
            xytext=(mb, y),
            arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.6},
        )
        ax.text(acc + 0.015, y + 0.22, f"{acc:.0%}", fontsize=9.5, color=INK, fontweight="bold")
        ax.text(mb - 0.015, y + 0.22, f"{mb:.0%}", fontsize=9.5, color=SLATE, ha="right")
    ax.set_yticks(range(len(attrs)), [ATTR_LABEL[a] for a in attrs[::-1]])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Accuracy on hidden labels (20% hidden)")
    ax.grid(axis="y", visible=False)
    fig.text(
        0.01,
        -0.03,
        "Grey = majority-class guess, teal = label propagation; one dot per "
        "campus, arrow = medians.",
        fontsize=8.5,
        color=SLATE,
    )
    meds = s["labelprop_median_over_schools"]
    yr, dm, mj = meds["year"], meds["dorm"], meds["major"]
    ax.set_title(
        f"The friendship graph alone recovers a hidden class year {yr['accuracy']:.0%} of the time "
        f"(majority guess: {yr['majority_baseline']:.0%});\n"
        f"dorm {dm['accuracy']:.0%} (vs {dm['majority_baseline']:.0%}), "
        f"major only {mj['accuracy']:.0%}, gender barely moves",
        pad=10,
    )
    _save(fig, path)


def communities_fig(res: dict[str, pd.DataFrame], s: dict, path: Path) -> None:
    comm = res["communities"]
    wide = comm.pivot_table(index=["school", "nodes"], columns="attribute", values="ari")
    wide = wide.reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    year_wins = wide["year"] >= wide["dorm"]
    ax.scatter(
        wide.loc[year_wins, "year"],
        wide.loc[year_wins, "dorm"],
        s=28,
        color=TEAL,
        alpha=0.8,
        linewidths=0,
        label="communities follow class year",
    )
    ax.scatter(
        wide.loc[~year_wins, "year"],
        wide.loc[~year_wins, "dorm"],
        s=28,
        color=AMBER,
        alpha=0.9,
        linewidths=0,
        label="communities follow dorm",
    )
    lim = max(wide["year"].max(), wide["dorm"].max()) + 0.05
    ax.plot([0, lim], [0, lim], color=SLATE, lw=1, ls="--")
    dorm_led = wide.loc[~year_wins].sort_values("dorm", ascending=False)
    for i, row in enumerate(dorm_led.itertuples()):
        # Alternate label sides: Caltech and Rice sit on top of each other.
        offset = (8, 6) if i % 2 == 0 else (8, -12)
        ax.annotate(
            row.school,
            (row.year, row.dorm),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            color=INK,
        )
    ax.set_xlim(-0.02, lim)
    ax.set_ylim(-0.02, lim)
    ax.set_xlabel("ARI between Louvain communities and class year")
    ax.set_ylabel("ARI between Louvain communities and dorm")
    ax.legend(loc="upper right")
    n, wins = len(wide), int(year_wins.sum())
    top2 = " and ".join(dorm_led["school"].head(2))
    ax.set_title(
        f"Communities track class year on {wins} of {n} campuses;\n"
        f"{n - wins} follow dorms instead, most clearly {top2}",
        pad=10,
    )
    _save(fig, path)


def structure_fig(res: dict[str, pd.DataFrame], s: dict, path: Path) -> None:
    st = res["structure"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.scatter(st["nodes"], st["mean_degree"], s=26, color=TEAL, alpha=0.8, linewidths=0)
    ax.set_xscale("log")
    for name in ["Caltech36", "Texas84", "Penn94", "MIT8"]:
        row = st[st["school"] == name]
        if len(row):
            ax.annotate(
                name,
                (row["nodes"].iloc[0], row["mean_degree"].iloc[0]),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=9,
                color=INK,
            )
    ax.set_xlabel("Campus size (nodes, log scale)")
    ax.set_ylabel("Mean number of friends")
    st_s = s["structure"]
    ratio = st_s["nodes_max"] / st_s["nodes_min"]
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_title(
        f"Campuses differ {ratio:.0f}x in size, but the mean number of friends only ranges "
        f"{st_s['mean_degree_min']:.0f}-{st_s['mean_degree_max']:.0f}",
        pad=10,
        fontsize=12,
    )
    _save(fig, path)


def draw_all(results_dir: str | Path, figures_dir: str | Path) -> None:
    _style()
    results_dir, figures_dir = Path(results_dir), Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    res = load_results(results_dir)
    s = json.loads((results_dir / "summary.json").read_text())
    hero_assortativity(res, s, figures_dir / "hero_assortativity.png")
    erratum(res, s, figures_dir / "erratum.png")
    link_prediction(res, s, figures_dir / "link_prediction.png")
    label_propagation(res, s, figures_dir / "label_propagation.png")
    communities_fig(res, s, figures_dir / "communities.png")
    structure_fig(res, s, figures_dir / "structure.png")
    print(f"Wrote figures to {figures_dir}")
