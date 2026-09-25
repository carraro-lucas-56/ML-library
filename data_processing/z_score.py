import sys
import numpy as np

def z_score_norm(data: np.ndarray) -> np.ndarray:
    sigma = np.std(data)
    mean = np.mean(data)

    return np.array((data-mean)/sigma)

def print_comparison(data, data_norm) -> None:
    for value, value2 in zip(data, data_norm):
        print(f"{value:<4} {value2:.3f}")

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

        print_comparison(sample, z_score_norm(sample))
