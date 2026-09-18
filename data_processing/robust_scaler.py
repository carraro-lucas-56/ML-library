import random
import sys

def median(data: list[int]) -> float:
    # Must receive the data sorted

    n = len(data)

    if n % 2 == 0:
        return (data[n//2 - 1] + data[n//2]) / 2 
    else:
        return data[(n-1)//2]

def computeIQR(data: list[int]) -> float:
    # Must receive the data sorted
    
    n = len(data)

    if n % 2 == 0:
        Q1 = median(data[:n//2])
        Q3 = median(data[n//2:])
    else:
        Q1 = median(data[:(n-1)//2])
        Q3 = median(data[n//2 + 1:])

    return Q3-Q1

def robust_scaler(data: list[int]) -> list[float]:
    data_sorted = sorted(data)

    IQR = computeIQR(data_sorted)
    med = median(data_sorted)

    return [(x-med)/IQR for x in data]

def print_comparison(data, data_norm):
    
    for x, x2 in zip(data, data_norm):
        print(f"{x:<3}  {x2:.3f}")

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        upper_bound = int(sys.argv[1])
        size = int(sys.argv[2])

        sample = [random.randint(1,upper_bound) for _ in range(size)]
        
        if len(sys.argv) == 4:
            outlier = int(sys.argv[3])
            sample.append(outlier)

        print_comparison(sample,robust_scaler(sample))

