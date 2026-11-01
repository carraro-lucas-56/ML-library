# ML Library

Classic machine learning algorithms implemented by hand with NumPy, as a way to learn how they work. This is a learning project, so the code favours clarity over speed and features, and isn't meant for production.

## What's implemented

| Area | Algorithm | File |
| --- | --- | --- |
| Clustering | K-Means | `clustering/k-means.py` |
| Clustering | DBSCAN | `clustering/dbscan.py` |
| Scaling | Min-max normalization | `data_processing/min_max_norm.py` |
| Scaling | Z-score normalization | `data_processing/z_score.py` |
| Scaling | Robust scaler | `data_processing/robust_scaler.py` |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scikit-learn pytest
```

The tests use scikit-learn and pytest. `clustering/dbscan.py` also imports scikit-learn, so it's needed to run DBSCAN.

## Usage

```python
import numpy as np
from clustering.dbscan import DBSCAN

data = np.random.default_rng(0).random((100, 2))
labels = DBSCAN(min_samples=5, eps=0.1).fit_predict(data)  # -1 marks noise
```

`KMeans` has a `fit` / `predict` interface and takes `k`, `max_iter` and `tol`. After `fit`, the results are in `model.centroids` and `model.inertia`. The hyphen in `k-means.py` rules out a normal `import`, so run the file directly or load it with `importlib`, as `tests/test_kmeans.py` does.

## Tests

```bash
python -m pytest tests
```

The tests compare results against scikit-learn where it has an equivalent, and also cover edge cases such as empty input, a single point and duplicate points.

## How AI is used

The model code is written by hand, because writing it is the point of this project. AI is never used to write the model implementations.

AI is used mostly for:

- **Writing tests:** developing the pytest suite, including the comparisons against scikit-learn and the edge cases.
- **Giving tips:** offering general advice on implementations and explaining in plain words what a failing test reveals, leaving the fix to me.

## Known limitations

K-Means is deliberately simple compared to scikit-learn:

- It starts from randomly chosen points, with no k-means++ and no restarts, so a run can settle on a poor solution.
- It raises an error when a cluster ends up empty instead of recovering.
- It has no `random_state` parameter. Use `np.random.seed` for reproducible runs.
