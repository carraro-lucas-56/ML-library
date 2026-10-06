import numpy as np

class DBSCAN:
    def __init__(self, min_samples: int, eps: float) -> None:
        # We define a core sample as being a sample in the dataset such that 
        # there exist min_samples other samples within a distance of eps

        # A cluster is therefore a set of core samples, each close to each other 
        # (measured by some distance measure) and a set of non-core samples that 
        # are close to a core sample (but are not themselves core samples). 
        
        self.min_sample = min_samples
        self.eps = eps

    def fit(self, data: np.ndarray) -> None:
        n = data.shape[0]
        labels = np.full(n, -1)

        all_idxs = np.array(range(n))
        
        idxs = np.full(1, 0, dtype=int)
        visited = np.zeros(n, dtype=bool)
        curr_label = 0

        while visited.sum() < n:
            # Points to be visited in the next interaction
            new_idxs = np.array([], dtype=int)

            for i in idxs:
                # (dim,) - (n, dim) = (n,dim)
                # norm((n,dim), axis=1) = (n,)
                dists = np.linalg.norm(data[i] - data, axis=1)

                neighboors = np.flatnonzero(dists <= self.eps)

                # Case where the point is a core sample
                if neighboors.size >= self.min_sample:
                    # Expands the current core sample
                    # Make sure we don't recompute any point 
                    new_idxs = np.concatenate((new_idxs, 
                                               neighboors[~visited[neighboors] &
                                                          ~np.isin(neighboors,new_idxs) & 
                                                          ~np.isin(neighboors,idxs)]))

                    # Assign a label to all density reachable points
                    labels[neighboors] = curr_label

                visited[i] = True

            idxs = new_idxs
            
            if idxs.size == 0 and visited.sum() < n:
                curr_label += 1 # must update the cluster only when we didn't start with a no core sample point
                idxs = np.random.choice(all_idxs[~visited], size=1)

        return labels

if __name__ == "__main__":
    n = 10
    m = 1000

    rng = np.random.default_rng()
    data = rng.random((m,n))

    model = DBSCAN(min_samples=10, eps=0.3)
    model.fit(data)