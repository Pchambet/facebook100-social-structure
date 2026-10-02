"""Self-contained static report (`site/index.html`) with interactive Plotly charts.

Every number in the narrative is read from `results/summary.json` or the result tables,
so the page cannot drift from the pipeline output.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from fb100.figures import ATTR_LABEL, ATTR_ORDER, METHOD_LABEL
from fb100.pipeline import load_results

TEAL, AMBER, SLATE = "#0d9488", "#d97706", "#64748b"
PLOTLY_JS = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"


def _campus(name: str) -> str:
    """Facebook100 file identifiers carry a numeric suffix (Caltech36); prose drops it."""
    return re.sub(r"\d+$", "", name)


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _not_year_led(not_year_led: dict[str, str], comm: pd.DataFrame) -> str:
    """', and dorm / house on Rice, Caltech, ...': campuses grouped by their leading
    attribute, strongest match first; '' when every campus follows class year."""
    ari = comm.set_index(["school", "attribute"])["ari"]
    groups: dict[str, list[str]] = {}
    for school, attr in not_year_led.items():
        groups.setdefault(attr, []).append(school)
    parts = [
        f"{ATTR_LABEL[attr].lower()} on "
        + _join([_campus(x) for x in sorted(schools, key=lambda x: -ari[x, attr])])
        for attr, schools in groups.items()
    ]
    return f", and {'; '.join(parts)}" if parts else ""


def _layout(fig: go.Figure, height: int = 420, **kwargs) -> go.Figure:
    kwargs.setdefault("showlegend", False)
    fig.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, system-ui, sans-serif", "color": SLATE, "size": 13},
        hoverlabel={"font": {"family": "Inter, system-ui, sans-serif"}},
        **kwargs,
    )
    grid = {"gridcolor": "rgba(100,116,139,0.18)", "zerolinecolor": "rgba(100,116,139,0.45)"}
    fig.update_xaxes(**grid, automargin=True)
    fig.update_yaxes(**grid, automargin=True)
    return fig


def _div(fig: go.Figure, div_id: str) -> str:
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=div_id,
        config={"displayModeBar": False, "responsive": True},
    )


def chart_assortativity(asr: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for attr in ATTR_ORDER[::-1]:
        d = asr[asr["attribute"] == attr].dropna(subset=["r"])
        fig.add_trace(
            go.Box(
                x=d["r"],
                y=[ATTR_LABEL[attr]] * len(d),
                name=ATTR_LABEL[attr],
                orientation="h",
                boxpoints="all",
                jitter=0.5,
                pointpos=0,
                marker={"size": 5, "color": TEAL if attr in {"year", "dorm"} else SLATE},
                line={"color": SLATE, "width": 1},
                fillcolor="rgba(0,0,0,0)",
                text=d["school"],
                hovertemplate="%{text}<br>r = %{x:.3f}<extra></extra>",
            )
        )
    return _layout(fig, 460, xaxis_title="assortativity r")


def chart_structure(st: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=st["nodes"],
            y=st["mean_degree"],
            mode="markers",
            marker={"size": 8, "color": TEAL, "opacity": 0.8},
            text=st["school"],
            customdata=st[["transitivity", "density"]],
            hovertemplate=(
                "%{text}<br>%{x:,} accounts<br>mean degree %{y:.1f}"
                "<br>transitivity %{customdata[0]:.3f}<br>density %{customdata[1]:.4f}<extra></extra>"
            ),
        )
    )
    ticks = [1000, 2000, 5000, 10000, 20000, 40000]
    return _layout(
        fig,
        380,
        xaxis={
            "type": "log",
            "title": "nodes (log)",
            "tickvals": ticks,
            "ticktext": [f"{t // 1000}k" for t in ticks],  # short enough not to tilt on phones
        },
        yaxis_title="mean degree",
    )


def chart_linkpred(full: pd.DataFrame) -> go.Figure:
    mean = full.groupby("method")[["auc_sampled", "r_precision", "precision_at_100"]].mean()
    order = mean.sort_values("r_precision").index
    labels = [METHOD_LABEL[m] for m in order]
    fig = go.Figure()
    for col, name, color in [
        ("auc_sampled", "sampled AUC", "rgba(100,116,139,0.45)"),
        ("precision_at_100", "precision@100", AMBER),
        ("r_precision", "R-precision", TEAL),
    ]:
        fig.add_trace(
            go.Bar(
                x=mean.loc[order, col],
                y=labels,
                name=name,
                orientation="h",
                marker_color=color,
                hovertemplate="%{y}<br>" + name + " = %{x:.3f}<extra></extra>",
            )
        )
    return _layout(
        fig,
        400,
        barmode="group",
        showlegend=True,
        legend={"orientation": "h", "y": -0.18},
        xaxis={"range": [0, 1], "title": "mean score"},
    )


def chart_labelprop(lp: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = {"year": TEAL, "dorm": AMBER, "major": SLATE, "gender": "#94a3b8"}
    for attr, color in colors.items():
        d = lp[lp["attribute"] == attr]
        fig.add_trace(
            go.Scatter(
                x=d["majority_baseline"],
                y=d["accuracy"],
                mode="markers",
                name=ATTR_LABEL[attr],
                marker={"size": 7, "color": color, "opacity": 0.8},
                text=d["school"],
                hovertemplate=(
                    ATTR_LABEL[attr] + " - %{text}<br>majority guess %{x:.0%}"
                    "<br>label propagation %{y:.0%}<extra></extra>"
                ),
            )
        )
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"color": SLATE, "dash": "dash"})
    return _layout(
        fig,
        440,
        showlegend=True,
        legend={"orientation": "h", "y": -0.2},
        xaxis={"range": [0, 1], "tickformat": ".0%", "title": "majority-class accuracy"},
        yaxis={"range": [0, 1], "tickformat": ".0%", "title": "label propagation accuracy"},
    )


def chart_communities(comm: pd.DataFrame) -> go.Figure:
    wide = comm.pivot_table(index="school", columns="attribute", values="ari").reset_index()
    year_wins = wide["year"] >= wide["dorm"]
    fig = go.Figure(
        go.Scatter(
            x=wide["year"],
            y=wide["dorm"],
            mode="markers",
            marker={"size": 8, "color": [TEAL if w else AMBER for w in year_wins]},
            text=wide["school"],
            hovertemplate="%{text}<br>ARI year %{x:.2f}<br>ARI dorm %{y:.2f}<extra></extra>",
        )
    )
    top = float(max(wide["year"].max(), wide["dorm"].max())) + 0.05
    fig.add_shape(type="line", x0=0, y0=0, x1=top, y1=top, line={"color": SLATE, "dash": "dash"})
    return _layout(
        fig,
        420,
        xaxis={"range": [-0.02, top], "title": "ARI communities vs class year"},
        yaxis={"range": [-0.02, top], "title": "ARI communities vs dorm"},
    )


CSS = """
:root { --bg:#ffffff; --ink:#0f172a; --muted:#64748b; --rule:#e2e8f0; --accent:#0d9488;
        --card:#f8fafc; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --bg:#0b1120; --ink:#e2e8f0; --muted:#94a3b8;
        --rule:#1e293b; --accent:#2dd4bf; --card:#111827; }
}
:root[data-theme="dark"] { --bg:#0b1120; --ink:#e2e8f0; --muted:#94a3b8; --rule:#1e293b;
        --accent:#2dd4bf; --card:#111827; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:16px/1.65 Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width:860px; margin:0 auto; padding:48px 16px 64px; }
h1 { font-size:2rem; line-height:1.2; margin:0 0 8px; letter-spacing:-0.01em; }
h2 { font-size:1.3rem; margin:48px 0 8px; padding-top:16px; border-top:1px solid var(--rule); }
p, li { color:var(--ink); }
.lede { color:var(--muted); font-size:1.1rem; margin:0 0 24px; }
.meta { color:var(--muted); font-size:0.9rem; }
a { color:var(--accent); }
.kpis { display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:12px;
        margin:24px 0; }
.kpi { background:var(--card); border:1px solid var(--rule); border-radius:10px; padding:14px; }
.kpi b { display:block; font-size:1.6rem; color:var(--accent); line-height:1.2; }
.kpi span { color:var(--muted); font-size:0.88rem; }
.takeaway { font-weight:600; }
.table-wrap { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:0.92rem; margin:12px 0; }
th, td { text-align:right; padding:6px 8px; border-bottom:1px solid var(--rule); }
th:first-child, td:first-child { text-align:left; }
th { color:var(--muted); font-weight:600; }
.chart { margin:8px 0 4px; }
.note { color:var(--muted); font-size:0.88rem; }
footer { margin-top:56px; color:var(--muted); font-size:0.9rem; }
"""


def _table(frame: pd.DataFrame) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in frame.columns)
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
        for row in frame.itertuples(index=False)
    )
    return (
        f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
    )


def build(results_dir: str | Path, out_path: str | Path) -> None:
    results_dir, out_path = Path(results_dir), Path(out_path)
    res = load_results(results_dir)
    s = json.loads((results_dir / "summary.json").read_text())
    a = s["assortativity"]
    lp = s["labelprop_median_over_schools"]
    full = s["linkpred_full_ranking_small_schools"]
    best = max(full, key=lambda m: full[m]["r_precision"])
    comm = s["communities"]
    year_wins = comm["best_attribute_counts"].get("year", 0)
    exceptions = _not_year_led(comm["not_year_led"], res["communities"])
    cn_aucs = [v["auc_sampled"] for m, v in full.items() if m != "preferential_attachment"]
    st = s["structure"]

    erratum = pd.DataFrame(
        [
            [ATTR_LABEL[k]]
            + [f"{a[v]['mean']:.3f}" for v in (f"{k} (legacy label)", f"{k} (missing counted)", k)]
            for k in ("year", "dorm")
        ],
        columns=[
            "Attribute",
            "First version",
            "Column order fixed",
            "Missing values also excluded",
        ],
    )
    lp_table = pd.DataFrame(
        [
            [
                ATTR_LABEL[k],
                f"{lp[k]['accuracy']:.0%}",
                f"{lp[k]['majority_baseline']:.0%}",
                f"{lp[k]['lift'] * 100:+.0f} pts",
            ]
            for k in ["year", "dorm", "gender", "major"]
        ],
        columns=["Attribute", "Label propagation", "Majority guess", "Median lift"],
    )
    lp_full = pd.DataFrame(
        [
            [
                METHOD_LABEL[m],
                f"{v['auc_sampled']:.3f}",
                f"{v['precision_at_100']:.2f}",
                f"{v['r_precision']:.3f}",
                f"{v['average_precision']:.3f}",
            ]
            for m, v in sorted(full.items(), key=lambda kv: -kv[1]["r_precision"])
        ],
        columns=["Method", "Sampled AUC", "Precision@100", "R-precision", "Average precision"],
    )

    charts = {
        "assort": _div(chart_assortativity(res["assortativity"]), "assort"),
        "structure": _div(chart_structure(res["structure"]), "structure"),
        "linkpred": _div(chart_linkpred(res["linkpred_full"]), "linkpred"),
        "labelprop": _div(chart_labelprop(res["labelprop"]), "labelprop"),
        "communities": _div(chart_communities(res["communities"]), "communities"),
    }
    base = full[best]["base_rate"]
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Facebook100 Social Structure</title>
<meta name="description" content="Homophily, link prediction, label propagation and communities
in 100 US campus friendship networks (Facebook, 2005).">
<link rel="icon" href="data:,">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap"
 rel="stylesheet">
<script src="{PLOTLY_JS}"></script>
<style>{CSS}</style>
</head>
<body>
<main>
<p class="meta">Network science report &middot; Facebook100 (September 2005)</p>
<h1>What organises friendship on a campus?</h1>
<p class="lede">{s["schools"]} complete campus friendship graphs, {s["nodes_total"]:,} accounts
and {s["edges_total"]:,} friendships: which attributes friends share, how predictable new
ties are, and whether the network's communities follow dorms or class years.</p>

<div class="kpis">
  <div class="kpi"><b>{a["year"]["median"]:.2f}</b><span>median class-year assortativity,
  the strongest of 7 attributes</span></div>
  <div class="kpi"><b>{a["dorm"]["median"]:.2f}</b><span>median dorm assortativity
  (the first version's median: {a["dorm (legacy label)"]["median"]:.3f})</span></div>
  <div class="kpi"><b>{lp["year"]["accuracy"]:.0%}</b><span>hidden class years recovered
  from the graph alone (majority guess {lp["year"]["majority_baseline"]:.0%})</span></div>
  <div class="kpi"><b>{full[best]["r_precision"]:.0%}</b><span>of top-ranked candidate
  pairs are real hidden friendships, despite AUC {full[best]["auc_sampled"]:.2f}</span></div>
</div>

<h2>1. Who befriends whom</h2>
<p>Newman's assortativity coefficient r measures how much more often friends share an
attribute than random mixing would predict (0 = no preference). Missing values are
excluded, as in Traud et al. (2012).</p>
<div class="chart">{charts["assort"]}</div>
<p class="takeaway">Class year dominates on {s["top_attribute_counts"].get("year", 0)} of
{s["schools"]} campuses (median r = {a["year"]["median"]:.3f}), followed by status and dorm
(median {a["dorm"]["median"]:.3f}). Major ({a["major"]["median"]:.3f}), gender
({a["gender"]["median"]:.3f}) and high school ({a["high_school"]["median"]:.3f}) barely
structure the network; degree assortativity is weak
({a["degree"]["median"]:.3f}).</p>

<h3>Erratum</h3>
<p>The first version of this project read <code>local_info</code> with six columns instead
of the documented seven (it skipped "second major / minor") and counted missing values as a
category. What it called "year" was the dorm, and what it called "dorm" was the minor. The
table recomputes the first version, then fixes one error at a time, in the same run (mean r
over {s["schools"]} campuses).</p>
{_table(erratum)}
<p class="note">Consequence: the first version's conclusion, that dorms barely matter, was an
artefact.
Dorms do matter (mean r {a["dorm"]["mean"]:.3f}, not {a["dorm (legacy label)"]["mean"]:.3f});
class year matters {a["year"]["mean"] / a["dorm"]["mean"]:.1f} times more.</p>

<h2>2. Size, density and clustering</h2>
<div class="chart">{charts["structure"]}</div>
<p class="takeaway">Campuses range from {st["nodes_min"]:,} to {st["nodes_max"]:,} accounts,
but the mean number of friends only ranges {st["mean_degree_min"]:.0f}-{st["mean_degree_max"]:.0f}
(median {st["mean_degree_median"]:.0f}). Density therefore falls with size (correlation of
logs {st["corr_log_size_log_density"]:.2f}), while transitivity stays around
{st["transitivity_median"]:.2f}, about {st["clustering_over_density_median"]:.0f} times the
density, the signature of a clustered, non-random network.</p>

<h2>3. Predicting hidden friendships</h2>
<p>10% of edges are hidden at random (5 splits on each of the 10 smallest campuses).
The common benchmark (AUC against random non-edges) is compared with the realistic task:
rank <em>every</em> unlinked pair, as a "people you may know" feature would.</p>
<div class="chart">{charts["linkpred"]}</div>
{_table(lp_full)}
<p class="takeaway">The four common-neighbour scores reach AUC {min(cn_aucs):.3f} to
{max(cn_aucs):.3f} on the easy benchmark;
on the full ranking the best ({METHOD_LABEL[best]}) gets {full[best]["r_precision"]:.1%} of
its top-R pairs right (R = the number of hidden friendships), about
{s["linkpred_best_lift_over_random"]} times the {base:.1%} base rate. The original claim of 96%
precision@100 (Adamic-Adar, 12 smallest campuses) came from ranking hidden edges against ten
times as many random non-edges, not against every candidate.</p>

<h2>4. Recovering missing attributes</h2>
<p>Harmonic label propagation on every campus, 20% of known labels hidden, compared with
always guessing the most frequent value.</p>
<div class="chart">{charts["labelprop"]}</div>
{_table(lp_table)}
<p class="takeaway">The graph alone recovers class year far above the baseline and dorms
substantially. Major stays hard ({lp["major"]["accuracy"]:.0%}) and gender gains only
{lp["gender"]["lift"] * 100:.0f} points, consistent with their weak assortativity.</p>

<h2>5. Do communities follow dorms or years?</h2>
<p>Louvain communities on the {comm["schools"]} campuses with at most 10,000 accounts,
compared with each attribute by the adjusted Rand index (0 = chance). Each dot is a campus:
teal where its communities match class year at least as well as dorm, orange where they
follow the dorm.</p>
<div class="chart">{charts["communities"]}</div>
<p class="takeaway">Communities follow class year on {year_wins} of {comm["schools"]}
campuses{exceptions}. Caltech is the
case highlighted by Traud et al.: its communities match the residential houses
(ARI {comm["caltech_ari"]["dorm"]:.2f}) and not the class year
(ARI {comm["caltech_ari"]["year"]:.2f}).</p>

<h2>Method notes and limitations</h2>
<ul>
<li>Data: Facebook100, a single snapshot from September 2005, intra-school links only.
Attributes are self-reported and partly missing; missing values are excluded per
attribute.</li>
<li>Assortativity is descriptive: it does not separate homophily (choosing similar friends)
from shared context (living in the same dorm, taking the same classes).</li>
<li>Link prediction hides edges uniformly at random, which is easier than predicting
future ties. Full-ranking metrics are computed on the 10 smallest campuses only (dense
n x n matrices).</li>
<li>Louvain is run once per campus (seed 0) and only on campuses with at most 10,000 nodes.</li>
</ul>

<footer>Built by <a href="https://github.com/Pchambet">Pierre Chambet</a> &mdash; decision
science for operations under uncertainty. Code and data access:
<a href="https://github.com/Pchambet/facebook100-social-structure">repository</a>.
Data: A. L. Traud, P. J. Mucha, M. A. Porter, <em>Social Structure of Facebook Networks</em>,
Physica A (2012).</footer>
</main>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page)
    print(f"Wrote {out_path}")
