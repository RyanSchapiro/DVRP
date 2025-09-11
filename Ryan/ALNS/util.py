import numpy as np
import random
import sys
import os


import numpy as np

def load_data(filename):
    """
    Loads instance data from a file with customer coordinates and demands.
    The file format is expected to be:
    NAME: ...
    VEHICLES: ...
    CAPACITY: ...
    CUSTNO XCOORD YCOORD DEMAND
    ...
    """
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
    """
    Splits customers into static and dynamic with Poisson arrival times.
    """
    # Use a local RandomState to not interfere with global numpy state
    # and to be controlled by the seed.
    rng = np.random.RandomState(seed)
    random_gen = random.Random(seed)

    customers = list(range(1, dim))  # exclude depot
    random_gen.shuffle(customers)
    split = int(static_ratio * len(customers))
    static = customers[:split]
    dynamic = customers[split:]
    arrivals = np.cumsum(rng.exponential(scale=1/lambda_rate, size=len(dynamic))).astype(int)
    # Ensure all arrival times are at least 1, since the simulation starts at t=1.
    # A customer with arrival time 0 would otherwise be missed.
    arrivals = np.maximum(1, arrivals)
    return set(static), {c: t for c, t in zip(dynamic, arrivals)}