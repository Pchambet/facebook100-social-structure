"""End-to-end analysis: every number in the README and the report comes from here.

Per-school work (structure, assortativity, sampled link-prediction AUC, label
propagation, Louvain) runs in a small process pool; the full-ranking link-prediction
benchmark and the label-propagation curves run on the ten smallest campuses, where an
n x n score matrix fits comfortably in memory.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from fb100 import communities, homophily, labelprop, linkpred, structure
from fb100.io import ATTRIBUTES, MISSING, list_schools, load_school

LP_ATTRIBUTES = ("dorm", "year", "major", "gender")
COMMUNITY_MAX_NODES = 10_000  # NetworkX Louvain on larger campuses costs minutes each
N_SMALL = 10
SEEDS = (0, 1, 2, 3, 4)
LP_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)

# The first version of this project read local_info with a 6-column map that skipped the
# minor column, and counted missing values (0) as a category. Kept to document the fix.
LEGACY_COLUMNS = {"year (legacy label)": 4, "dorm (legacy label)": 3}


def _per_school(path: Path) -> dict[str, list[dict]]:
    school = load_school(path)
    a, attrs = school.adjacency, school.attributes
    out: dict[str, list[dict]] = {k: [] for k in ("structure", "assort", "auc", "lp", "comm")}
    out["structure"].append(structure.summarize(school))

    for name in ATTRIBUTES:
        values = attrs[name].to_numpy()
        out["assort"].append(
            {
                "school": school.name,
                "nodes": school.n_nodes,
                "attribute": name,
                "r": homophily.attribute_assortativity(a, values),
                "labelled_share": homophily.labelled_share(values),
            }
        )
    out["assort"].append(
        {
            "school": school.name,
            "nodes": school.n_nodes,
            "attribute": "degree",
            "r": out["structure"][0]["degree_assortativity"],
            "labelled_share": 1.0,
        }
    )
    raw = attrs.to_numpy()
    for label, col in LEGACY_COLUMNS.items():
        out["assort"].append(
            {
                "school": school.name,
                "nodes": school.n_nodes,
                "attribute": label,
                "r": homophily.attribute_assortativity(a, raw[:, col], missing=None),
                "labelled_share": 1.0,
            }
        )

    for row in linkpred.evaluate(a, seed=0, full_ranking=False):
        out["auc"].append({"school": school.name, "nodes": school.n_nodes, **row})

    for name in LP_ATTRIBUTES:
        result = labelprop.evaluate(a, attrs[name].to_numpy(), hidden_fraction=0.2, seed=0)
        out["lp"].append({"school": school.name, "nodes": school.n_nodes, "attribute": name})
        out["lp"][-1].update(result)

    if school.n_nodes <= COMMUNITY_MAX_NODES:
        partition, modularity = communities.louvain(a, seed=0)
        for row in communities.agreement(partition, attrs[list(LP_ATTRIBUTES)]):
            out["comm"].append(
                {
                    "school": school.name,
                    "nodes": school.n_nodes,
                    "communities": int(partition.max() + 1),
                    "modularity": modularity,
                    **row,
                }
            )
    print(f"  {school.name:<16} n={school.n_nodes:>6}", flush=True)
    return out


def _small_school_benchmarks(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    lp_rows, curve_rows = [], []
    for path in paths:
        school = load_school(path)
        for seed in SEEDS:
            for row in linkpred.evaluate(school.adjacency, seed=seed, ks=(100, 1000)):
                lp_rows.append({"school": school.name, "nodes": school.n_nodes, **row})
        for name in LP_ATTRIBUTES:
            values = school.attributes[name].to_numpy()
            for fraction in LP_FRACTIONS:
                for seed in SEEDS[:3]:
                    result = labelprop.evaluate(school.adjacency, values, fraction, seed)
                    curve_rows.append(
                        {
                            "school": school.name,
                            "attribute": name,
                            "hidden_fraction": fraction,
                            "seed": seed,
                            **result,
                        }
                    )
        print(f"  {school.name:<16} benchmarks done", flush=True)
    return pd.DataFrame(lp_rows), pd.DataFrame(curve_rows)


def run(data_dir: str | Path, out_dir: str | Path, jobs: int = 3, limit: int | None = None):
    """Run every analysis and write the result tables to `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = list_schools(data_dir)
    if not paths:
        raise FileNotFoundError(f"no school files in {data_dir}; run `make data` first")
    sizes = {p: load_school(p).n_nodes for p in paths}
    paths.sort(key=sizes.get)
    if limit:
        paths = paths[:limit]

    print(f"Per-school analyses on {len(paths)} campuses ({jobs} workers)")
    tables: dict[str, list[dict]] = {}
    # Largest graphs first so the pool does not end on one long straggler.
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for part in pool.map(_per_school, paths[::-1]):
            for key, rows in part.items():
                tables.setdefault(key, []).extend(rows)

    print(f"Full-ranking benchmarks on the {N_SMALL} smallest campuses")
    lp_full, lp_curve = _small_school_benchmarks(paths[:N_SMALL])

    frames = {
        "structure": pd.DataFrame(tables["structure"]),
        "assortativity": pd.DataFrame(tables["assort"]),
        "linkpred_auc": pd.DataFrame(tables["auc"]),
        "labelprop": pd.DataFrame(tables["lp"]),
        "communities": pd.DataFrame(tables["comm"]),
        "linkpred_full": lp_full,
        "labelprop_curve": lp_curve,
    }
    for name, frame in frames.items():
        sort_cols = [c for c in ("nodes", "school", "attribute", "method", "seed") if c in frame]
        frame.sort_values(sort_cols).to_csv(out_dir / f"{name}.csv", index=False)
    summary = summarize_results(frames)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {len(frames)} tables and summary.json to {out_dir}")
    return summary


def load_results(out_dir: str | Path) -> dict[str, pd.DataFrame]:
    names = [
        "structure",
        "assortativity",
        "linkpred_auc",
        "labelprop",
        "communities",
        "linkpred_full",
        "labelprop_curve",
    ]
    return {n: pd.read_csv(Path(out_dir) / f"{n}.csv") for n in names}


def _r(x: float, nd: int = 3) -> float:
    return round(float(x), nd)


def summarize_results(frames: dict[str, pd.DataFrame]) -> dict:
    """Headline numbers quoted in the README and the report."""
    st, asr = frames["structure"], frames["assortativity"]
    auc, lp, comm = frames["linkpred_auc"], frames["labelprop"], frames["communities"]
    full = frames["linkpred_full"]

    by_attr = asr.groupby("attribute")["r"]
    assort = {
        a: {
            "median": _r(g.median()),
            "mean": _r(g.mean()),
            "q25": _r(g.quantile(0.25)),
            "q75": _r(g.quantile(0.75)),
            "schools": int(g.notna().sum()),
        }
        for a, g in by_attr
    }
    real = asr[asr["attribute"].isin(ATTRIBUTES)].dropna(subset=["r"])
    top = real.loc[real.groupby("school")["r"].idxmax(), "attribute"].value_counts()

    full_mean = full.groupby("method")[
        ["auc_sampled", "average_precision", "r_precision", "precision_at_100", "base_rate"]
    ].mean()
    lp_summary = (
        lp.assign(lift=lp["accuracy"] - lp["majority_baseline"])
        .groupby("attribute")[["accuracy", "majority_baseline", "lift"]]
        .median()
    )
    best_comm = comm.loc[comm.groupby("school")["ari"].idxmax()]
    caltech = comm[comm["school"] == "Caltech36"].set_index("attribute")["ari"]

    def corr_log(x: str, y: str) -> float:
        return _r(np.corrcoef(np.log(st[x]), np.log(st[y]))[0, 1])

    return {
        "schools": len(st),
        "nodes_total": int(st["nodes"].sum()),
        "edges_total": int(st["edges"].sum()),
        "structure": {
            "nodes_min": int(st["nodes"].min()),
            "nodes_max": int(st["nodes"].max()),
            "mean_degree_median": _r(st["mean_degree"].median(), 1),
            "mean_degree_min": _r(st["mean_degree"].min(), 1),
            "mean_degree_max": _r(st["mean_degree"].max(), 1),
            "corr_log_size_log_mean_degree": corr_log("nodes", "mean_degree"),
            "corr_log_size_log_density": corr_log("nodes", "density"),
            "transitivity_median": _r(st["transitivity"].median()),
            "clustering_over_density_median": _r((st["transitivity"] / st["density"]).median(), 1),
        },
        "assortativity": assort,
        "top_attribute_counts": {k: int(v) for k, v in top.items()},
        "linkpred_auc_all_schools_median": {
            m: _r(g.median()) for m, g in auc.groupby("method")["auc_sampled"]
        },
        "linkpred_full_ranking_small_schools": {
            m: {k: _r(v, 4) for k, v in row.items()} for m, row in full_mean.iterrows()
        },
        "labelprop_median_over_schools": {
            a: {k: _r(v) for k, v in row.items()} for a, row in lp_summary.iterrows()
        },
        "communities": {
            "schools": int(comm["school"].nunique()),
            "best_attribute_counts": {
                k: int(v) for k, v in best_comm["attribute"].value_counts().items()
            },
            "median_ari": {a: _r(g.median()) for a, g in comm.groupby("attribute")["ari"]},
            "caltech_ari": {a: _r(v) for a, v in caltech.items()},
        },
        "missing_code": MISSING,
    }


def default_jobs() -> int:
    return int(os.environ.get("FB100_JOBS", "3"))
