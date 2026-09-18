import numpy as np


class KMeans():
    def __init__(self, k: int) -> None:
        self.k = k

    def fit(self, data: np.ndarray):
        n = len(data)
        k = self.k
        
        chosen_idxs = np.random.choice(data.shape[0], size=k, replace=False)
        centroids = data[chosen_idxs]
    
        for _ in range(5):        
            # 1. Compute Euclidean distance from every point to every centroid using broadcasting
            # shape change example -> (100, 1, 5) - (k, 5) = (100, k, 5), norm((100, k, 5)) = (100,k)
            distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    
            # 2. Get the index of the closest centroid for each of the 100 points
            map_point_centroid = np.argmin(distances, axis=1)
            
            centroids = np.zeros((k,data.shape[1]))
            cnt = np.zeros((k,))
            
            for j in range(n):
                centroid_num = map_point_centroid[j]
                centroids[centroid_num] += data[j]        
                cnt[centroid_num] += 1
    
            # handle empty centroid problem here
            if 0 in cnt:
                pass
    
            # compute new centroids
            centroids /= cnt[:, np.newaxis]

        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)

        self.inertia = np.sum(np.min(distances, axis=1))
        self.centroids = centroids 

    def predict(self, data: np.ndarray) -> np.ndarray[float]:
        if data.shape[-1] != self.centroids.shape[-1]:
            raise

        distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
        

if __name__ == "__main__":
    n = 10
    m = 1000
    k = 3

    rng = np.random.default_rng()
    data = rng.random((m,n))

    model = KMeans(k=k)
    model.fit(data)

    # print(model.predict(data))
    print(model.inertia)