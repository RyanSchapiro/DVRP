import numpy as np
import random
import sys
import os


import numpy as np

def load_data(filename):
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    vehicle_count = int(lines[1].split(':')[1].strip())
    vehicle_capacity = int(lines[2].split(':')[1].strip())

    customer_data = lines[4:]
    coords = []
    demands = []
    for line in customer_data:
        parts = line.split()
        coords.append([float(parts[1]), float(parts[2])])
        demands.append(int(parts[3]))

    coords = np.array(coords)
    demands = np.array(demands, dtype=int)
    return coords, demands, vehicle_capacity, vehicle_count

def Poisson(dim, static_ratio=0.7, lambda_rate=0.4, seed=None):

    rng = np.random.RandomState(seed)
    random_gen = random.Random(seed)

    customers = list(range(1, dim)) 
    random_gen.shuffle(customers)
    split = int(static_ratio * len(customers))
    static = customers[:split]
    dynamic = customers[split:]
    arrivals = np.cumsum(rng.exponential(scale=1/lambda_rate, size=len(dynamic))).astype(int)
    arrivals = np.maximum(1, arrivals)
    return set(static), {c: t for c, t in zip(dynamic, arrivals)}