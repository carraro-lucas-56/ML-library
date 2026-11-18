import time
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering as SkAgglomerativeClustering

class AgglomerativeClustering():
    def __init__(self, n_clusters: int = 2) -> None:
        self.n_clusters = n_clusters

    @staticmethod
    def _compute_dist(dists: np.ndarray, 
                      label_1: int, 
                      label_2: int , 
                      map_point_label: np.ndarray) -> float:
        """
        Formula used (single-linkage) -> d(A,B) = min {a in A, b in B} d(a,b)
        """

        points_cluster_1 = np.where(map_point_label == label_1)[0]
        points_cluster_2 = np.where(map_point_label == label_2)[0] 
        
        return dists[points_cluster_1][:, points_cluster_2].min()

    def fit(self, data: np.ndarray) -> None:
        n, dim = data.shape

        # distances between each data point individually
        # (n, 1 ,dim) - (n, dim) = (n, n, dim)
        # norm((n, n, dim), axis=2) = (n,n)
        dists = np.linalg.norm(data[:, np.newaxis] - data, axis=2) 

        # Map each point to its cluster, initially each point is part of a single point cluster
        map_point_label = np.array(range(n))

        for _ in range(n-1):
            labels = np.unique(map_point_label)
            smaller_dist = 10000000

            # Variables to store the clusters to be merged
            merge_1 = merge_2 = 0

            i, j = np.triu_indices(len(labels), k=1)

            for lbl_1, lbl_2 in np.stack([labels[i], labels[j]], axis=-1):
                
                if lbl_1 == lbl_2:
                    continue

                # Compute the distance between the current clusters
                curr_dist = self._compute_dist(dists, lbl_1, lbl_2, map_point_label)
                
                if curr_dist < smaller_dist:
                    merge_1 = lbl_1
                    merge_2 = lbl_2
                    smaller_dist = curr_dist

            # Merge the clusters
            map_point_label[map_point_label == merge_2] = merge_1

        print(map_point_label)

if __name__ == "__main__":
    n = 10
    m = 1000

    rng = np.random.default_rng()
    data = rng.random((m,n))
    data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.5, random_state=0)

    # model = AgglomerativeClustering()
    model = SkAgglomerativeClustering()

    start = time.perf_counter()

    print(100*"=")
    print(time.perf_counter() - start, "seconds")

