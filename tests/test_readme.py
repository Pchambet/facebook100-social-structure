"""The README quotes the committed results; these checks keep the two from drifting."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
SUMMARY = json.loads((ROOT / "results" / "summary.json").read_text())


def test_linked_files_exist():
    for target in re.findall(r"\]\(((?!https?://)[^)#]+)\)", README):
        assert (ROOT / target).exists(), target


def test_assortativity_numbers():
    a = SUMMARY["assortativity"]
    for attr in ["year", "status", "dorm", "major", "gender", "high_school"]:
        assert f"{a[attr]['median']:.3f}" in README, attr
    assert f"effect of {a['dorm (legacy label)']['mean']:.3f}" in README
    assert f"dorm effect is {a['dorm (missing counted)']['mean']:.3f}" in README
    assert f"excluded\n  it is {a['dorm']['mean']:.3f}" in README
    # Cost of counting missing values as a category, once the columns are fixed.
    costs = [a[k]["mean"] - a[f"{k} (missing counted)"]["mean"] for k in ("dorm", "year")]
    assert f"another {min(costs):.2f}-{max(costs):.2f} in r" in README
    assert f"on {SUMMARY['top_attribute_counts']['year']} of {SUMMARY['schools']}" in README


def test_link_prediction_numbers():
    full = SUMMARY["linkpred_full_ranking_small_schools"]
    assert f"only {full['resource_allocation']['r_precision']:.0%} of the top-ranked" in README
    assert f"it is {full['adamic_adar']['precision_at_100']:.0%}" in README
    for row in full.values():
        assert f"| {row['auc_sampled']:.3f} | {row['precision_at_100']:.2f} |" in README


def test_label_propagation_and_communities():
    lp = SUMMARY["labelprop_median_over_schools"]
    assert f"class year {lp['year']['accuracy']:.0%} of the time" in README
    assert f"year: {lp['year']['majority_baseline']:.0%})" in README
    assert f"dorm {lp['dorm']['accuracy']:.0%} of the time" in README
    comm = SUMMARY["communities"]
    assert f"year on {comm['best_attribute_counts']['year']} of {comm['schools']}" in README
    assert f"Caltech ({comm['caltech_ari']['dorm']:.2f})" in README
    others = comm["schools"] - comm["best_attribute_counts"]["year"]
    assert f"On {others} of the {comm['schools']} campuses analysed" in README
    assert f"on the other {comm['best_attribute_counts']['year']}, class year" in README
