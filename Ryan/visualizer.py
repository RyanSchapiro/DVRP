import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
import re

# --- Configuration ---
INSTANCE_NAMES = ["c1_100.txt", "r1_100.txt", "rc1_100.txt"]
# NOTE: This path assumes the script is run from the 'Ryan' directory.
# The instance files are expected to be in 'Dataset/reduced_instances'.
# The original Solomon instance files (e.g., c101.txt) are processed by
# 'Dataset/reducer.py' to create files like 'c101_100.txt'.
# We assume 'c101_100.txt' has been renamed to 'c1_100.txt' to match
# the naming convention in the results files.
SOLVER_RESULTS_PATH = "optimal_solutions.json"

ALGORITHM_STYLES = {
    "Optimal": {"color": "black", "label": "Optimal", "linestyle": "--"},
    "ALNS": {"color": "blue", "label": "ALNS", "linestyle": "-"},
    "Clarke-Wright": {"color": "green", "label": "Clarke-Wright", "linestyle": "-"},
    "SAC": {"color": "#fb00c5", "label": "SAC", "linestyle": "-"},
}

# --- Data Loading ---

def load_data(filename):
    """
    Loads instance data from a file with customer coordinates and demands.
    This function is adapted from ALNS/util.py.
    """
    if not os.path.exists(filename):
        print(f"Error: Instance file not found at '{filename}'")
        print(f"Please ensure the instance file exists and the path is correct.")
        print("You may need to run 'Dataset/reducer.py' and rename the output file (e.g., 'c101_100.txt' to 'c1_100.txt').")
        return None, None, None, None

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

# Removed load_best_route_from_csv and load_best_route_from_rl_csv as they are no longer used.
def parse_solver_routes(result_text):
    """Parses routes from the PyVRP result string."""
    routes = []
    lines = result_text.split('\n')
    
    # Find the start of the routes section
    try:
        routes_start_index = next(i for i, line in enumerate(lines) if line.strip().lower() == 'routes')
        lines_to_parse = lines[routes_start_index + 2:]
    except StopIteration:
        lines_to_parse = lines

    for line in lines_to_parse:
        if not line.strip().startswith("Route #"):
            continue
            
        try:
            route_part = line.split(':')[-1].strip()
            if not route_part: continue
            
            # Split by space and convert to int
            customers = [int(c.strip()) for c in route_part.split()]
            if customers:
                # Add depot to start and end for plotting
                routes.append([0] + customers + [0])
        except (ValueError, IndexError):
            print(f"Warning: Could not parse route from line: '{line}'")
            continue
    return routes

def load_routes_from_solver_json(filepath, instance_name):
    """Load optimal routes for a specific instance from the solver's JSON file."""
    if not os.path.exists(filepath):
        print(f"Warning: Solver file not found at '{filepath}'. Skipping.")
        return None
    try:
        with open(filepath, "r") as f:
            all_solutions = json.load(f)
        
        if instance_name in all_solutions:
            solution_data = all_solutions[instance_name]
            if solution_data.get('success'):
                result_text = solution_data['result']
                return parse_solver_routes(result_text)
            else:
                print(f"Warning: Solver did not find a successful solution for '{instance_name}'.")
                return None
        else:
            print(f"Warning: Instance '{instance_name}' not found in solver file '{filepath}'.")
            return None
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error reading solver file '{filepath}': {e}")
        return None

# --- Visualization ---

def plot_customer_locations(instance_data):
    """
    Plots customer coordinates for multiple instances to compare clustering.
    """
    num_instances = len(instance_data)
    if num_instances == 0:
        print("No customer data to plot.")
        return

    fig, axes = plt.subplots(1, num_instances, figsize=(8 * num_instances, 8.5), 
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()

    for i, (instance_name, data) in enumerate(instance_data.items()):
        ax = axes[i]
        coords = data["coords"]
        
        # Plot customers and depot
        depot_coords = coords[0]
        customer_coords = coords[1:]
        ax.scatter(customer_coords[:, 0], customer_coords[:, 1], c='black', s=20, label='Customers')
        ax.scatter(depot_coords[0], depot_coords[1], c='red', marker='*', s=200, label='Depot', zorder=5)

        ax.set_title(f"Customers: {instance_name.replace('.txt', '')}", fontsize=14)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_aspect('equal', adjustable='box')

    fig.supxlabel("X Coordinate", fontsize=14, y=0.03)
    fig.supylabel("Y Coordinate", fontsize=14, x=0.01)
    plt.tight_layout(rect=[0.03, 0.05, 1, 0.95])
    plt.show()

def plot_optimal_routes(instance_data):
    """
    Plots optimal routes for multiple instances to compare routing solutions.
    """
    num_instances = len(instance_data)
    if num_instances == 0:
        print("No route data to plot.")
        return

    fig, axes = plt.subplots(1, num_instances, figsize=(8 * num_instances, 8.5), 
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()

    style = ALGORITHM_STYLES["Optimal"]
    for i, (instance_name, data) in enumerate(instance_data.items()):
        ax = axes[i]
        coords = data["coords"]
        routes = data["routes"]

        # Plot customers and depot
        depot_coords = coords[0]
        customer_coords = coords[1:]
        ax.scatter(customer_coords[:, 0], customer_coords[:, 1], c='black', s=20, label='Customers')
        ax.scatter(depot_coords[0], depot_coords[1], c='red', marker='*', s=200, label='Depot', zorder=5)

        # Plot routes
        if routes:
            for j, route in enumerate(routes):
                route_coords = coords[route]
                label = style["label"] if j == 0 else None
                ax.plot(route_coords[:, 0], route_coords[:, 1],
                        color=style["color"],
                        label=label,
                        linestyle=style.get("linestyle", "-"),
                        alpha=0.9, zorder=4, linewidth=2)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_aspect('equal', adjustable='box')

    fig.supxlabel("X Coordinate", fontsize=14, y=0.03)
    fig.supylabel("Y Coordinate", fontsize=14, x=0.01)
    plt.tight_layout(rect=[0.03, 0.05, 1, 0.95])
    plt.show()

# --- Main Execution ---

def main():
    """Main function to load data and generate visualization."""
    print("Visualizing customer locations and optimal routes for instance comparison...")

    instance_data_to_plot = {}
    instance_file_path_template = os.path.join("..", "Dataset", "files", "{}")

    for instance_name in INSTANCE_NAMES:
        print(f"\nProcessing {instance_name}...")

        # 1. Load customer coordinates
        instance_file_path = instance_file_path_template.format(instance_name)
        coords, _, _, _ = load_data(instance_file_path)
        if coords is None:
            continue

        # 2. Load optimal routes
        routes = load_routes_from_solver_json(SOLVER_RESULTS_PATH, instance_name)
        if routes is None:
            print(f"Warning: Could not load optimal routes for {instance_name}. Plotting without routes.")
            routes = [] # Still plot customers if routes are missing

        instance_data_to_plot[instance_name] = {
            "coords": coords,
            "routes": routes
        }

    if not instance_data_to_plot:
        print("No instance data was successfully loaded. Nothing to plot.")
        return

    # 3. Plot the locations and routes in separate figures
    plot_customer_locations(instance_data_to_plot)
    plot_optimal_routes(instance_data_to_plot)


if __name__ == "__main__":
    main()
