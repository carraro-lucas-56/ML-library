from math import sqrt
import sys, random

def std_deviation(data: list) -> float:
    n = len(data)
    mean = sum(data)/n
    variance = sum((x-mean)**2 for x in data) / n

    return sqrt(variance) 

def z_score_norm(data) -> list[float]:
    sigma = std_deviation(data)
    mean = sum(data)/len(data)

    return [(x-mean)/sigma for x in data]

def print_comparison(data, data_norm) -> None:
    for value, value2 in zip(data, data_norm):
        print(f"{value:<4} {value2:.3f}")

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        upper_bound = int(sys.argv[1])
        size = int(sys.argv[2])

        sample = [random.randint(1,upper_bound) for _ in range(size)]

        if len(sys.argv) == 4:
            outlier = int(sys.argv[3])
            sample.append(outlier)

        print_comparison(sample,z_score_norm(sample))
