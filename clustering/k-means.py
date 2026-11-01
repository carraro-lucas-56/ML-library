import numpy as np

class KMeans():
    def __init__(self, k: int, 
                 max_iter: int = 300, 
                 tol: float =1e-4) -> None:
        
        self.num_centroids = k
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, data: np.ndarray) -> None:
        # n is the number of data points and dim the dimension of those points
        n, dim = data.shape
        k = self.num_centroids
        
        # Randomly selecting the initial centroids
        chosen_idxs = np.random.choice(n, size=k, replace=False)
        centroids = data[chosen_idxs]
        old_centroids = centroids
    
        for _ in range(self.max_iter):        
            # Compute Euclidean distance from every point to every centroid using broadcasting
            # (n, 1, dim) - (k, dim) = (n, k, dim)
            # norm((n, k, dim), axis=2) = (n, k)
            distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    
            # Get the index of the closest centroid for each of the n points
            # (n, k) -> (n,)
            map_point_centroid = np.argmin(distances, axis=1)
            
            centroids = np.zeros((k, dim))
            cnt = np.zeros((k,))
            
            for j in range(n):
                centroid_idx = map_point_centroid[j]
                centroids[centroid_idx] += data[j]        
                cnt[centroid_idx] += 1
    
            # handle empty centroid problem here
            if 0 in cnt:
                raise
    
            # compute new centroids
            # (k, dim) /= (k, 1) = (k, dim)
            centroids /= cnt[:, np.newaxis]

            # Condition to check if the centroids are moving considerably
            if np.allclose(centroids, old_centroids, atol=self.tol):
                break

            old_centroids=centroids

        # (n, 1 ,dim) - (k, dim) = (n, k, dim)
        # norm((n, k, dim), axis=2) = (n,k) 
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)

        # min((n,k), axis=1) = (n,)
        # sum((n,)) = float
        self.inertia = np.sum(np.min(distances, axis=1)**2)
        
        self.centroids = centroids 

    def predict(self, data: np.ndarray) -> np.ndarray[float]:
        if data.shape[-1] != self.centroids.shape[-1]:
            raise Exception("data has wrong dimensions")

        # (n, 1, dim) - (k, dim) = (n, k, dim)
        # norm((n, k, dim), axis=2) = (n, k)
        distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)

        # (n,)
        return np.argmin(distances, axis=1)
        

if __name__ == "__main__":
    n = 10
    m = 1000
    k = 30

    rng = np.random.default_rng()
    data = rng.random((m,n))
    

    model = KMeans(k=k)
    model.fit(data)

    print(model.predict(data))
    print(model.inertia)

