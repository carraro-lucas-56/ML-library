import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN as SkDBSCAN

class DBSCAN:
    def __init__(self, min_samples: int, eps: float) -> None:
        # We define a core sample as being a sample in the dataset such that 
        # there exist min_samples other samples within a distance of eps

        # A cluster is therefore a set of core samples, each close to each other 
        # (measured by some distance measure) and a set of non-core samples that 
        # are close to a core sample (but are not themselves core samples). 
        
        self.min_sample = min_samples
        self.eps = eps

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        n = data.shape[0]
        labels = np.full(n, -1)

        all_idxs = np.array(range(n))

        curr_points = np.zeros(n, dtype=bool)
        next_points = np.zeros(n, dtype=bool)
        visited     = np.zeros(n, dtype=bool)

        curr_points[0] = True
        curr_label = 0
        num_visited = 0

        while num_visited < n:

            for i in np.flatnonzero(curr_points):
                # (dim,) - (n, dim) = (n,dim) -> norm((n,dim), axis=1) = (n,)
                dists = np.linalg.norm(data[i] - data, axis=1)

                neighbors = np.flatnonzero(dists <= self.eps)

                # Case where the point is a core sample
                if neighbors.size >= self.min_sample:
                    # Expands the current core sample, make sure we don't recompute any point 
                    next_points[neighbors] |= (~visited[neighbors] & ~curr_points[neighbors])

                    # Assign a label to all density reachable points
                    labels[neighbors] = curr_label

                if visited[i] == False:
                    num_visited += 1
                    visited[i] = True

            
            curr_points[:] = next_points
            next_points[:] = False
            
            if curr_points.sum() == 0 and num_visited < n:
                # Only increase the label if the last interactions produced a valid cluster
                if curr_label in labels:
                    curr_label += 1

                # Pick an unvisited point to restart the process
                curr_points[all_idxs[~visited][0]] = True

        return labels

if __name__ == "__main__":
    n = 10
    m = 1000

    rng = np.random.default_rng()
    data = rng.random((m,n))
    data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.5, random_state=0)

    model = DBSCAN(min_samples=10, eps=0.3)
    model2 = SkDBSCAN(min_samples=10, eps=0.3)

    arr1 = model.fit_predict(data)
    arr2 = model2.fit_predict(data)

    ours = np.unique(arr1)
    skl = np.unique(arr2)

    print(ours)
    print(skl)

    d = {}
    for label in arr1:
        if str(label) in d.keys():
            d[str(label)] += 1
        else:
            d[str(label)] = 1
    
    d2 = {}
    for label in arr2:
        if str(label) in d2.keys():
            d2[str(label)] += 1
        else:
            d2[str(label)] = 1

    print(d)
    print(d2)

