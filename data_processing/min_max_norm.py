import sys
import numpy as np

def min_max_norm(data: np.ndarray, lower_bound: float, upper_bound: float) -> np.ndarray:
    max_value = np.max(data)
    min_value = np.min(data)

    return np.array(lower_bound + (data-min_value)/(max_value-min_value)*(upper_bound-lower_bound))  

def print_comparison(data, data_norm) -> None:
    for value, value2 in zip(data, data_norm):
        print(f"{value:<4} {value2:.5f}")

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

        print_comparison(sample,
                         min_max_norm(sample,
                                      lower_bound=0,
                                      upper_bound=1))
    
