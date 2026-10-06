import numpy as np
import pytest
from sklearn.cluster import DBSCAN as SkDBSCAN
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from clustering.dbscan import DBSCAN

def test_matches_sklearn_on_separated_blobs():
    # Well-separated blobs: no ambiguous border points, so results must match sklearn exactly
    # (up to label numbering, which ARI ignores).
    data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.5, random_state=0)
    ours = DBSCAN(min_samples=5, eps=0.5).fit(data)
    ref = SkDBSCAN(min_samples=5, eps=0.5).fit_predict(data)

    assert adjusted_rand_score(ref, ours) == pytest.approx(1.0)
    np.testing.assert_array_equal(ours == -1, ref == -1)  # same noise points


def test_noise_points_match_sklearn():
    # Noise is deterministic in DBSCAN (border points are not), so noise sets must match.
    rng = np.random.default_rng(42)
    data = rng.random((500, 2))
    ours = DBSCAN(min_samples=8, eps=0.05).fit(data)
    ref = SkDBSCAN(min_samples=8, eps=0.05).fit_predict(data)

    np.testing.assert_array_equal(ours == -1, ref == -1)


def test_all_noise_when_eps_too_small():
    data = np.random.default_rng(0).random((100, 2))
    labels = DBSCAN(min_samples=5, eps=1e-6).fit(data)
    assert np.all(labels == -1)


def test_single_cluster_when_eps_huge():
    data = np.random.default_rng(0).random((100, 2))
    labels = DBSCAN(min_samples=5, eps=10).fit(data)
    assert np.unique(labels).size == 1 and labels[0] != -1