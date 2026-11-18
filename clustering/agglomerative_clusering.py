import time
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering as SkAgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

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
        point_dists = np.linalg.norm(data[:, np.newaxis] - data, axis=2) 
        
        cluster_dists = np.linalg.norm(data[:, np.newaxis] - data, axis=2) 

        # Map each point to its cluster, initially each point is part of a single point cluster
        map_point_label = np.array(range(n))

        for k in range(n-self.n_clusters):
            labels = np.unique(map_point_label)
            smaller_dist = 10000000

            # Variables to store the clusters to be merged
            candidate_1 = candidate_2 = 0

            i, j = np.triu_indices(len(labels), k=1)

            for lbl_1, lbl_2 in np.stack([labels[i], labels[j]], axis=-1):
                
                if lbl_1 == lbl_2:
                    continue

                # Compute the distance between the current clusters
                if k != 0 and (lbl_1 in [merge_1, merge_2] or lbl_2 in [merge_1, merge_2]):
                    curr_dist = self._compute_dist(point_dists, lbl_1, lbl_2, map_point_label)
                    cluster_dists[lbl_1][lbl_2] = curr_dist
                    cluster_dists[lbl_2][lbl_1] = curr_dist
                else:
                    curr_dist = cluster_dists[lbl_1][lbl_2]
                
                if curr_dist < smaller_dist:
                    candidate_1 = lbl_1
                    candidate_2 = lbl_2
                    smaller_dist = curr_dist

            merge_1 = candidate_1
            merge_2 = candidate_2

            # Merge the clusters
            map_point_label[map_point_label == merge_2] = merge_1

        self.labels_ = map_point_label

if __name__ == "__main__":
    n = 10
    m = 1000

    rng = np.random.default_rng()
    data = rng.random((m,n))
    data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.5, random_state=0)

    # start = time.perf_counter()

    my_model = AgglomerativeClustering(n_clusters=3)
    my_model.fit(data)

    sk_model = SkAgglomerativeClustering(n_clusters=3, linkage="single")
    sk_model.fit(data)

    print(adjusted_rand_score(my_model.labels_, sk_model.labels_))  # 1.0 = identical clustering

