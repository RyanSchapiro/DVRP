import json
import os
import matplotlib.pyplot as plt
import numpy as np

files = ["c1_100.txt", "r1_100.txt", "rc1_100.txt"]
solutions = "optimal_solutions.json"

def load_data(file):
    if not os.path.exists(file):
        return None, None
    
    with open(file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    data = lines[4:]
    coords = []
    for line in data:
        parts = line.split()
        coords.append([float(parts[1]), float(parts[2])])
    
    return np.array(coords), None

def parse_routes(text):
    routes = []
    lines = text.split('\n')
    
    for line in lines:
        if not line.strip().startswith("Route #"):
            continue
        try:
            route_part = line.split(':')[-1].strip()
            if route_part:
                customers = [int(c.strip()) for c in route_part.split()]
                if customers:
                    routes.append([0] + customers + [0])
        except:
            continue
    return routes

def load_routes(file, instance):
    if not os.path.exists(file):
        return None
    
    try:
        with open(file, "r") as f:
            data = json.load(f)
        
        if instance in data and data[instance].get('success'):
            return parse_routes(data[instance]['result'])
    except:
        pass
    return None

def plot_customers(data):
    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(8*n, 8))
    if n == 1:
        axes = [axes]
    
    for i, (name, coords) in enumerate(data.items()):
        ax = axes[i]
        depot = coords[0]
        customers = coords[1:]
        
        ax.scatter(customers[:, 0], customers[:, 1], c='black', s=20, label='Customers')
        ax.scatter(depot[0], depot[1], c='red', marker='*', s=200, label='Depot')
        ax.set_title(name.replace('.txt', ''))
        ax.legend()
        ax.grid(True, alpha=0.6)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

def plot_routes(data):
    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(8*n, 8))
    if n == 1:
        axes = [axes]
    
    for i, (name, info) in enumerate(data.items()):
        ax = axes[i]
        coords = info["coords"]
        routes = info["routes"]
        
        depot = coords[0]
        customers = coords[1:]
        
        ax.scatter(customers[:, 0], customers[:, 1], c='black', s=20, label='Customers')
        ax.scatter(depot[0], depot[1], c='red', marker='*', s=200, label='Depot')
        
        if routes:
            for j, route in enumerate(routes):
                route_coords = coords[route]
                label = 'Optimal' if j == 0 else None
                ax.plot(route_coords[:, 0], route_coords[:, 1], 
                       'k--', label=label, alpha=0.9, linewidth=2)
        
        ax.set_title(name.replace('.txt', ''))
        ax.legend()
        ax.grid(True, alpha=0.6)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

def main():
    data = {}
    path = os.path.join("..", "Dataset", "files", "{}")
    
    for file in files:
        coords, _ = load_data(path.format(file))
        if coords is None:
            continue
        
        routes = load_routes(solutions, file)
        if routes is None:
            routes = []
        
        data[file] = {"coords": coords, "routes": routes}
    
    if data:
        plot_customers({k: v["coords"] for k, v in data.items()})
        plot_routes(data)

if __name__ == "__main__":
    main()