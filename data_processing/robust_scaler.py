import sys
import numpy as np

def computeIQR(data: np.ndarray) -> float:
    # Must receive the data sorted
    
    n = len(data)

    if n % 2 == 0:
        Q1 = np.median(data[:n//2])
        Q3 = np.median(data[n//2:])
    else:
        Q1 = np.median(data[:(n-1)//2])
        Q3 = np.median(data[n//2 + 1:])

    return Q3-Q1

def robust_scaler(data: list[int]) -> list[float]:
    data_sorted = sorted(data)

    IQR = computeIQR(data_sorted)
    med = np.median(data_sorted)

    return np.array((data_sorted-med)/IQR) 

def print_comparison(data, data_norm):
    
    for x, x2 in zip(data, data_norm):
        print(f"{x:<3}  {x2:.3f}")

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        min_value = int(sys.argv[1])
        max_value = int(sys.argv[2])
        
        size = int(sys.argv[3])

        rng = np.random.default_rng()
        sample = (max_value-min_value) * rng.random((size,)) + min_value

        if len(sys.argv) == 5:
            outlier = int(sys.argv[4])
            sample[-1] = outlier

        print_comparison(sample,robust_scaler(sample))

