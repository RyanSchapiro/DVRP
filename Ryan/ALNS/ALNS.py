import sys
import copy
import time
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from itertools import permutations

from alns import ALNS
from alns.accept import RecordToRecordTravel
from alns.select import RouletteWheel
from alns.stop import MaxIterations
from util import load_data, Poisson

def compute_distance_matrix(coords):
    distances = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=-1)
    return distances.astype(int)

SEED = 89
STATIC_RATIO = 0.7
LAMBDA_RATE = 0.4
MAX_SIMULATION_TIME = 1000
REOPT_ITERATIONS = 3000
UNASSIGNED_PENALTY = 1000
DESTRUCTION_RATE = 0.2
MAX_STRING_REMOVALS = 2
MAX_STRING_SIZE = 6


rnd.seed(SEED)

class CvrpState:
    
    def __init__(self, routes, unassigned=None):
        self.routes = routes
        self.unassigned = unassigned or []

    def copy(self):
        return CvrpState(copy.deepcopy(self.routes), self.unassigned.copy())

    def objective(self):
        total_cost = sum(self._route_cost(route) for route in self.routes)
        vehicle_cost = len(self.routes) * 50  # Add vehicle costs
        total_cost += vehicle_cost + len(self.unassigned) * UNASSIGNED_PENALTY
        return total_cost
    @property
    def cost(self):
        return self.objective()

    def find_route(self, customer):
        for route in self.routes:
            if customer in route:
                return route
        raise ValueError(f"Customer {customer} not found in any route")

    def _route_cost(self, route):
        tour = [0] + route + [0]
        return sum(distance_matrix[tour[i]][tour[i+1]] for i in range(len(tour) - 1))


def random_removal(state, rng):
    destroyed = state.copy()
    assigned = [c for route in destroyed.routes for c in route]
    
    if not assigned:
        return destroyed
    
    num_to_remove = max(1, int(len(assigned) * DESTRUCTION_RATE))
    num_to_remove = min(num_to_remove, len(assigned))
    
    for customer in rng.choice(assigned, num_to_remove, replace=False).tolist():
        destroyed.unassigned.append(customer)
        route = destroyed.find_route(customer)
        route.remove(customer)
    
    return _remove_empty_routes(destroyed)

def string_removal(state, rng):
    destroyed = state.copy()
    if not state.routes:
        return destroyed
    
    avg_route_size = int(np.mean([len(route) for route in state.routes]))
    max_string_size = max(MAX_STRING_SIZE, avg_route_size)
    max_removals = min(len(state.routes), MAX_STRING_REMOVALS)
    destroyed_routes = []
    
    if n_customers > 1:
        center = rng.integers(1, n_customers + 1)
        for customer in _get_neighbors(center):
            if len(destroyed_routes) >= max_removals:
                break
            if customer in destroyed.unassigned:
                continue
            
            try:
                route = destroyed.find_route(customer)
                if route in destroyed_routes:
                    continue
                
                removed = _remove_string_from_route(route, customer, max_string_size, rng)
                destroyed.unassigned.extend(removed)
                destroyed_routes.append(route)
            except ValueError:
                continue
    
    return destroyed

def _get_neighbors(customer):
    distances = np.argsort(distance_matrix[customer])
    return [i for i in distances if i != 0]

def _remove_string_from_route(route, customer, max_size, rng):
    size = rng.integers(1, min(len(route), max_size) + 1)
    start = route.index(customer) - rng.integers(size)
    indices = [idx % len(route) for idx in range(start, start + size)]
    
    removed = []
    for idx in sorted(indices, reverse=True):
        removed.append(route.pop(idx))
    return removed

def _remove_empty_routes(state):
    state.routes = [route for route in state.routes if route]
    return state

def greedy_repair(state, rng):
    repaired = state.copy()
    rng.shuffle(repaired.unassigned)
    
    while repaired.unassigned:
        customer = repaired.unassigned.pop()
        route, idx = _find_best_insertion(customer, repaired)
        
        if route is not None:
            route.insert(idx, customer)
        else:
            repaired.routes.append([customer])
    
    return repaired

def _find_best_insertion(customer, state):
    best_cost, best_route, best_idx = None, None, None
    
    for route in state.routes:
        if not _can_insert(customer, route):
            continue
        
        for idx in range(len(route) + 1):
            cost = _insertion_cost(customer, route, idx)
            if best_cost is None or cost < best_cost:
                best_cost, best_route, best_idx = cost, route, idx
    
    return best_route, best_idx

def _can_insert(customer, route):
    total_demand = sum(demands[c] for c in route) + demands[customer]
    return total_demand <= capacity

def _insertion_cost(customer, route, idx):
    pred = 0 if idx == 0 else route[idx - 1]
    succ = 0 if idx == len(route) else route[idx]
    return (distance_matrix[pred][customer] + 
            distance_matrix[customer][succ] - 
            distance_matrix[pred][succ])

def main():
    global coords, demands, capacity, n_customers, distance_matrix
    
    # Load problem data
    folder = "../../dataset/files"
    filename = input("Enter filename (e.g. c1_25.txt): ")
    
    coords, demands, capacity, vehicle_count = load_data(f"{folder}/{filename}")
    n_customers = len(demands) - 1
    distance_matrix = compute_distance_matrix(coords)
    
    static_customers, dynamic_customers = Poisson(
        len(demands), STATIC_RATIO, LAMBDA_RATE, seed=SEED
    )
    
    print(f"Problem: {len(static_customers)} static, {len(dynamic_customers)} dynamic customers")
    
    # Initialize ALNS
    start_time = time.time()
    alns = ALNS(rnd)
    alns.add_destroy_operator(random_removal)
    alns.add_destroy_operator(string_removal)
    alns.add_repair_operator(greedy_repair)
    
    select = RouletteWheel([25, 5, 1, 0], 0.8, 1, 1)
    
    # Create initial solution
    print("\n--- Time 0: Building initial solution ---")
    initial_state = CvrpState(routes=[], unassigned=list(static_customers))
    current_solution = greedy_repair(initial_state, rnd)
    
    print(f"Initial cost: {current_solution.objective()}")
    print(f"Initial routes: {current_solution.routes}")
    
    # Run simulation
    for t in range(1, MAX_SIMULATION_TIME + 1):
        new_customers = [c for c, arrival_time in dynamic_customers.items() 
                        if arrival_time == t]
        
        if not new_customers:
            continue
        
        print(f"\n--- Time {t}: New customers {new_customers} ---")
        current_solution.unassigned.extend(new_customers)
        
        # Re-optimize with ALNS
        accept = RecordToRecordTravel.autofit(
            current_solution.objective(), 0.02, 0, REOPT_ITERATIONS
        )
        stop = MaxIterations(REOPT_ITERATIONS)
        
        result = alns.iterate(current_solution, select, accept, stop)
        current_solution = result.best_state
        
        print(f"Re-optimized cost: {current_solution.objective()}")
        print(f"Re-optimized routes: {current_solution.routes}")
        
        all_revealed = all(arrival_time <= t for c, arrival_time in dynamic_customers.items())
        if all_revealed and not current_solution.unassigned:
            print("\nAll customers revealed and routed")
            break
    
    final_time = time.time() - start_time   
    print("\n--- Final Results ---")
    
    
    objective = current_solution.objective()
    print(f"Total time: {final_time:.2f} seconds")
    print(f"Final cost: {objective}")
    print(f"Vehicles used: {len(current_solution.routes)}")
    
    for i, route in enumerate(current_solution.routes):
        print(f"  Vehicle {i+1}: {route}")
    
    if current_solution.unassigned:
        print(f"Unassigned: {current_solution.unassigned}")

if __name__ == "__main__":
    main()