"""The README quotes the committed results; these checks keep the two from drifting."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README_RAW = (ROOT / "README.md").read_text()
# Line breaks in the Markdown source carry no meaning: match on single spaces.
README = " ".join(README_RAW.split())
SUMMARY = json.loads((ROOT / "results" / "summary.json").read_text())


def test_linked_files_exist():
    for target in re.findall(r"\]\(((?!https?://)[^)#]+)\)", README_RAW):
        assert (ROOT / target).exists(), target


def test_assortativity_numbers():
    a = SUMMARY["assortativity"]
    for attr in ["year", "status", "dorm", "major", "gender", "high_school"]:
        assert f"{a[attr]['median']:.3f}" in README, attr
    assert f"effect of {a['dorm (legacy label)']['mean']:.3f}" in README
    assert f"dorm effect is {a['dorm (missing counted)']['mean']:.3f}" in README
    assert f"excluded it is {a['dorm']['mean']:.3f}" in README
    # Cost of counting missing values as a category, once the columns are fixed.
    costs = [a[k]["mean"] - a[f"{k} (missing counted)"]["mean"] for k in ("dorm", "year")]
    assert f"another {min(costs):.2f}-{max(costs):.2f} in r" in README
    assert f"on {SUMMARY['top_attribute_counts']['year']} of {SUMMARY['schools']}" in README


def test_link_prediction_numbers():
    full = SUMMARY["linkpred_full_ranking_small_schools"]
    assert f"only {full['resource_allocation']['r_precision']:.0%} of the top-ranked" in README
    assert f"it is {full['adamic_adar']['precision_at_100']:.0%}" in README
    assert f"{SUMMARY['linkpred_best_lift_over_random']} times the" in README
    for row in full.values():
        assert f"| {row['auc_sampled']:.3f} | {row['precision_at_100']:.2f} |" in README
    # Sampled AUC over all campuses, quoted as a range for the common-neighbour scores.
    auc = SUMMARY["linkpred_auc_all_schools_median"]
    cn = [v for m, v in auc.items() if m != "preferential_attachment"]
    assert f"median {min(cn):.2f}-{max(cn):.2f} for the common-neighbour scores" in README


def test_label_propagation_and_communities():
    lp = SUMMARY["labelprop_median_over_schools"]
    assert f"class year {lp['year']['accuracy']:.0%} of the time" in README
    assert f"year: {lp['year']['majority_baseline']:.0%})" in README
    assert f"dorm {lp['dorm']['accuracy']:.0%} of the time" in README
    assert f"recoverable from the graph ({lp['year']['accuracy']:.0%})" in README
    assert f"({lp['dorm']['accuracy']:.0%} vs {lp['dorm']['majority_baseline']:.0%})" in README
    assert f"major only partly ({lp['major']['accuracy']:.0%})" in README
    curve = SUMMARY["labelprop_curve_small_schools"]
    year = curve["median_accuracy"]["year"]
    assert (
        f"median over {curve['runs_per_point']} runs: {year['0.1']:.0%} with 10% hidden, "
        f"{year['0.9']:.0%} with 90% hidden"
    ) in README
    assert f"standard deviation is {curve['seed_sd_median'] * 100:.0f} points" in README
    comm = SUMMARY["communities"]
    assert f"year on {comm['best_attribute_counts']['year']} of {comm['schools']}" in README
    assert f"Caltech ({comm['caltech_ari']['dorm']:.2f})" in README
    others = comm["schools"] - comm["best_attribute_counts"]["year"]
    assert f"On {others} of the {comm['schools']} campuses analysed" in README
    assert f"on the other {comm['best_attribute_counts']['year']}, class year" in README
