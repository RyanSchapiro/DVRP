from pyvrp import Model
from pyvrp.stop import MaxRuntime
from pyvrp.plotting import plot_solution
import matplotlib.pyplot as plt
import numpy as np

def load_data(filename):
    """
    Loads instance data from a file with customer coordinates and demands.
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
    distance_matrix = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=-1)
    return coords, distance_matrix, demands, vehicle_capacity, vehicle_count

# ==== EDIT ====
filename = "../dataset/files/r1_100.txt"  # Replace with your data file
vehicle_fixed_cost = 50                # Set to your vehicle_weight
runtime_seconds = 20                   # Time to search for a solution
# ==============

coords, distance_matrix, demands, vehicle_capacity, num_vehicles = load_data(filename)
n_nodes = distance_matrix.shape[0]

m = Model()
m.add_vehicle_type(
    num_vehicles,
    capacity=[int(vehicle_capacity)],
    fixed_cost=vehicle_fixed_cost
)
depot = m.add_depot(x=int(coords[0][0]), y=int(coords[0][1]))

clients = []
for idx in range(1, n_nodes):
    clients.append(
        m.add_client(
            x=int(coords[idx][0]),
            y=int(coords[idx][1]),
            delivery=[int(demands[idx])]
        )
    )


for frm_idx, frm in enumerate(m.locations):
    for to_idx, to in enumerate(m.locations):
        distance = distance_matrix[frm_idx][to_idx]
        m.add_edge(frm, to, distance=int(distance))

print("Solving...")
res = m.solve(stop=MaxRuntime(runtime_seconds), display=True)

print(res)

# Plot solution
_, ax = plt.subplots(figsize=(8, 8))
plot_solution(res.best, m.data(), ax=ax)
plt.show()