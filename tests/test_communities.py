import numpy as np
import pandas as pd
import pytest

from fb100.communities import agreement, louvain


def test_louvain_recovers_planted_blocks_and_ignores_noise(sbm):
    a, blocks = sbm
    partition, modularity = louvain(a, seed=0)
    noise = np.random.default_rng(0).integers(1, 5, size=len(blocks))
    attrs = pd.DataFrame({"block": blocks, "noise": noise})
    rows = {r["attribute"]: r for r in agreement(partition, attrs)}
    assert modularity > 0.5
    assert rows["block"]["ari"] > 0.95
    assert rows["noise"]["ari"] == pytest.approx(0, abs=0.02)


def test_agreement_ignores_missing_values():
    partition = np.array([0, 0, 1, 1, 2])
    attrs = pd.DataFrame({"x": [3, 3, 4, 4, 0]})  # last node unlabelled
    assert agreement(partition, attrs)[0]["ari"] == pytest.approx(1.0)
