import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.cluster import KMeans as SkKMeans
from sklearn.datasets import load_iris, make_blobs
from sklearn.metrics import adjusted_rand_score

# The module's file name has a hyphen, so it can't be loaded with a regular import statement.
_path = Path(__file__).resolve().parent.parent / "clustering" / "k-means.py"
_spec = importlib.util.spec_from_file_location("k_means", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
KMeans = _module.KMeans


def fit_seeded(data, k, seed=0, **kwargs):
    # The model draws its initial centroids from NumPy's global random state.
    np.random.seed(seed)
    model = KMeans(k=k, **kwargs)
    model.fit(data)
    return model


def initial_centroids(data, k, seed):
    # Same draw that fit() makes first, so sklearn can start from identical centroids.
    np.random.seed(seed)
    return data[np.random.choice(data.shape[0], size=k, replace=False)]


def fit_sklearn_from(data, init, **kwargs):
    # n_init=1 and an explicit init turn off k-means++ and restarts, leaving plain Lloyd iterations.
    return SkKMeans(n_clusters=init.shape[0], init=init, n_init=1, algorithm="lloyd", **kwargs).fit(data)


def sort_rows(centroids):
    # Cluster numbering is arbitrary, so put centroids in a canonical order before comparing.
    return centroids[np.lexsort(centroids.T[::-1])]


def blobs(n_samples=300, centers=4, cluster_std=0.5, n_features=2, seed=0):
    data, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std,
                         n_features=n_features, random_state=seed)
    return data


DATASETS = {
    "separated_blobs": (blobs(), 4),
    "overlapping_blobs": (blobs(cluster_std=2.5), 4),
    "high_dim_blobs": (blobs(n_features=10, centers=5), 5),
    "uniform_noise": (np.random.default_rng(42).random((300, 3)), 6),
    "iris": (load_iris().data, 3),
}


# Comparison against sklearn.
# Default sklearn uses k-means++ and 10 restarts, and the model uses one random start, so the two
# can legitimately land in different local minima. Giving both the same starting centroids makes
# the comparison fair: Lloyd's algorithm is deterministic from there.

@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("name", DATASETS)
def test_converges_to_same_solution_as_sklearn_from_same_init(name, seed):
    data, k = DATASETS[name]
    ours = fit_seeded(data, k, seed=seed, tol=1e-10)
    ref = fit_sklearn_from(data, initial_centroids(data, k, seed), tol=0, max_iter=300)

    np.testing.assert_allclose(sort_rows(ours.centroids), sort_rows(ref.cluster_centers_), atol=1e-6)
    assert adjusted_rand_score(ref.labels_, ours.predict(data)) == pytest.approx(1.0)


@pytest.mark.parametrize("name", DATASETS)
def test_single_iteration_matches_sklearn(name):
    data, k = DATASETS[name]
    ours = fit_seeded(data, k, seed=0, max_iter=1)
    ref = fit_sklearn_from(data, initial_centroids(data, k, 0), max_iter=1)

    np.testing.assert_allclose(sort_rows(ours.centroids), sort_rows(ref.cluster_centers_))


def test_best_of_restarts_recovers_sklearn_clusters_on_separated_blobs():
    # Without k-means++, a single run may get stuck, so restart by hand like n_init does.
    data = blobs()
    runs = [fit_seeded(data, 4, seed=seed) for seed in range(10)]
    best = min(runs, key=lambda model: np.sum((data - model.centroids[model.predict(data)]) ** 2))
    ref = SkKMeans(n_clusters=4, n_init=10, random_state=0).fit(data)

    assert adjusted_rand_score(ref.labels_, best.predict(data)) == pytest.approx(1.0)


# Inertia

def test_inertia_is_sum_of_squared_distances():
    # One centroid at (0, 2), both points at distance 2: 2² + 2² = 8.
    data = np.array([[0.0, 0.0], [0.0, 4.0]])
    assert fit_seeded(data, 1).inertia == pytest.approx(8.0)


@pytest.mark.parametrize("name", DATASETS)
def test_inertia_matches_sklearn_from_same_init(name):
    data, k = DATASETS[name]
    ours = fit_seeded(data, k, tol=1e-10)
    ref = fit_sklearn_from(data, initial_centroids(data, k, 0), tol=0, max_iter=300)

    assert ours.inertia == pytest.approx(ref.inertia_)


# Properties of the algorithm

def test_k_equal_one_gives_the_mean():
    data = blobs()
    model = fit_seeded(data, 1)

    np.testing.assert_allclose(model.centroids, data.mean(axis=0, keepdims=True))
    assert np.all(model.predict(data) == 0)


def test_k_equal_n_gives_every_point_its_own_cluster():
    data = np.random.default_rng(0).random((20, 2))
    model = fit_seeded(data, 20)

    np.testing.assert_allclose(sort_rows(model.centroids), sort_rows(data))
    assert np.unique(model.predict(data)).size == 20
    assert model.inertia == pytest.approx(0.0)


@pytest.mark.parametrize("name", DATASETS)
def test_solution_is_a_fixed_point(name):
    # After convergence, each centroid must be the mean of the points assigned to it.
    data, k = DATASETS[name]
    model = fit_seeded(data, k, tol=1e-10)
    labels = model.predict(data)
    means = np.array([data[labels == i].mean(axis=0) for i in range(k)])

    np.testing.assert_allclose(model.centroids, means, atol=1e-8)


def test_predict_assigns_each_point_to_its_nearest_centroid():
    data = blobs()
    model = fit_seeded(data, 4)
    new_points = np.random.default_rng(1).uniform(data.min(axis=0), data.max(axis=0), size=(50, 2))
    distances = np.linalg.norm(new_points[:, None] - model.centroids, axis=2)

    np.testing.assert_array_equal(model.predict(new_points), distances.argmin(axis=1))


def test_predicting_a_centroid_returns_its_own_label():
    model = fit_seeded(blobs(), 4)
    np.testing.assert_array_equal(model.predict(model.centroids), np.arange(4))


def test_cluster_sizes_on_separated_blobs():
    # make_blobs splits samples evenly, so a correct clustering has exactly 75 points per cluster.
    data = blobs()
    labels = min(
        (fit_seeded(data, 4, seed=seed) for seed in range(10)),
        key=lambda model: np.sum((data - model.centroids[model.predict(data)]) ** 2),
    ).predict(data)

    np.testing.assert_array_equal(np.bincount(labels), [75, 75, 75, 75])


# Parameters and API

def test_max_iter_stops_early():
    data = DATASETS["overlapping_blobs"][0]
    one_step = fit_seeded(data, 4, seed=0, max_iter=1)
    converged = fit_seeded(data, 4, seed=0, tol=1e-10)

    assert not np.allclose(one_step.centroids, converged.centroids)


def test_huge_tol_stops_after_first_iteration():
    data = DATASETS["overlapping_blobs"][0]
    one_step = fit_seeded(data, 4, seed=0, max_iter=1)
    huge_tol = fit_seeded(data, 4, seed=0, tol=1e9)

    np.testing.assert_array_equal(huge_tol.centroids, one_step.centroids)


def test_same_seed_is_reproducible():
    data = blobs()
    first, second = fit_seeded(data, 4, seed=7), fit_seeded(data, 4, seed=7)

    np.testing.assert_array_equal(first.centroids, second.centroids)
    assert first.inertia == second.inertia


def test_refit_replaces_previous_solution():
    model = fit_seeded(blobs(), 4)
    new_data = np.random.default_rng(0).random((50, 3))
    model.fit(new_data)

    assert model.centroids.shape == (4, 3)
    assert model.predict(new_data).shape == (50,)


def test_fit_does_not_modify_input():
    data = blobs()
    original = data.copy()
    fit_seeded(data, 4)

    np.testing.assert_array_equal(data, original)


def test_integer_input():
    data = np.array([[0, 0], [0, 2], [10, 10], [10, 12]])
    model = fit_seeded(data, 1)

    np.testing.assert_allclose(model.centroids, [[5.0, 6.0]])


@pytest.mark.parametrize("dim", [1, 2, 3, 10, 50])
def test_output_shapes_for_different_dimensionalities(dim):
    data = blobs(n_samples=100, centers=3, n_features=dim)
    model = fit_seeded(data, 3)
    labels = model.predict(data)

    assert model.centroids.shape == (3, dim)
    assert labels.shape == (100,)
    assert set(np.unique(labels)) <= {0, 1, 2}


# Edge cases

def test_single_point():
    data = np.array([[1.0, 2.0]])
    model = fit_seeded(data, 1)

    np.testing.assert_array_equal(model.centroids, data)
    np.testing.assert_array_equal(model.predict(data), [0])
    assert model.inertia == 0


def test_predict_empty_input():
    model = fit_seeded(blobs(), 4)
    assert model.predict(np.empty((0, 2))).shape == (0,)


def test_predict_before_fit_raises():
    with pytest.raises(AttributeError):
        KMeans(k=2).predict(np.zeros((3, 2)))


def test_predict_with_wrong_dimension_raises():
    model = fit_seeded(blobs(), 4)
    with pytest.raises(Exception, match="wrong dimensions"):
        model.predict(np.zeros((5, 3)))


@pytest.mark.parametrize(
    "data, k",
    [
        pytest.param(np.empty((0, 2)), 2, id="empty_input"),
        pytest.param(np.zeros((3, 2)), 4, id="k_greater_than_n"),
        pytest.param(np.zeros((3, 2)), 0, id="k_zero"),
        pytest.param(np.zeros((3, 2)), -1, id="k_negative"),
        pytest.param(np.zeros(3), 1, id="one_dimensional_array"),
    ],
)
def test_invalid_input_raises(data, k):
    with pytest.raises(ValueError):
        fit_seeded(data, k)


def test_all_points_identical_with_one_cluster():
    data = np.full((10, 3), 5.0)
    model = fit_seeded(data, 1)

    np.testing.assert_array_equal(model.centroids, [[5.0, 5.0, 5.0]])
    assert model.inertia == 0


# The model doesn't recover from empty clusters (sklearn relocates a far away point instead), it
# raises. These tests describe that limitation; change them if empty clusters get handled.

def test_empty_cluster_raises_when_all_points_are_identical():
    with pytest.raises(RuntimeError):
        fit_seeded(np.full((10, 3), 5.0), 2)


def test_duplicate_points_either_fail_on_empty_cluster_or_find_the_two_groups():
    # Whether a run hits an empty cluster depends on whether both initial centroids are the same point.
    data = np.array([[0.0, 0.0]] * 5 + [[10.0, 10.0]] * 5)
    solved = 0
    for seed in range(20):
        try:
            model = fit_seeded(data, 2, seed=seed)
        except RuntimeError:
            continue
        solved += 1
        np.testing.assert_allclose(sort_rows(model.centroids), [[0.0, 0.0], [10.0, 10.0]])

    assert solved > 0
