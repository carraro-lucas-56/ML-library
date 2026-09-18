import random, sys

def min_max_norm(data: list[float], lower_bound: float, upper_bound: float) -> list[float]:
    max_value = max(data)
    min_value = min(data)

    return [lower_bound + (value-min_value)/(max_value-min_value)*(upper_bound-lower_bound) for value in data]

def print_comparison(data, data_norm) -> None:
    for value, value2 in zip(data, data_norm):
        print(f"{value:<4} {value2:.2f}")

if __name__ == "__main__":

    if len(sys.argv) > 1:
        upper_bound = int(sys.argv[1])
        size = int(sys.argv[2])

        sample = [random.randint(1,upper_bound) for _ in range(size)]

        if len(sys.argv) == 4:
            outlier = int(sys.argv[3])
            sample.append(outlier)

        print_comparison(sample,
                         min_max_norm(sample,
                                      lower_bound=1,
                                      upper_bound=10))
    
