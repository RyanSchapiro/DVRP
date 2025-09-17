import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import time

from util import load_data, Poisson

SEED = 89                   
STATIC_RATIO = 0.7        
LAMBDA_RATE = 0.4       
MAX_SIM_TIME = 1000         # Maximum simulation time steps
PENALTY = 1000              # Penalty cost for unassigned customers (capacity violations)

 # Random seed for reproducibility
rnd.seed(SEED)

def calc_distances(coords):
    """Calculate Euclidean distance matrix.
    
    Args:
        coords: numpy array of (x,y) coordinates for depot and customers
        
    Returns:
        Integer distance matrix where distances[i][j] = distance from i to j
    """
    distances = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=-1)
    return distances.astype(int)

class Solution:
    """VRP Solution State"""
    
    def __init__(self, routes, unassigned=None):
        """Initialize solution with routes and unassigned customers.
        
        Args:
            routes: List of routes, where each route is a list of customer IDs
            unassigned: List of customer IDs that couldn't be assigned to routes
        """
        self.routes = routes
        self.unassigned = unassigned if unassigned is not None else []
    
    def total_cost(self):
        """Calculate total solution cost including routing, vehicle, and penalty costs.
        
        Returns:
            Total cost = travel distance + vehicle fixed costs + unassigned penalties
        """
        # Sum travel costs for all routes
        total_distance = 0
        for route in self.routes:
            total_distance += self._route_cost(route)

        # Add fixed cost per vehicle (50 per vehicle) and penalty for unassigned
        vehicle_cost = len(self.routes) * 50
        penalty_cost = len(self.unassigned) * PENALTY
        return total_distance + penalty_cost + vehicle_cost
    
    def _route_cost(self, route):
        """Calculate total travel distance for a single route.
        
        Args:
            route: List of customer IDs in visit order
            
        Returns:
            Total distance starting and ending at depot (customer 0)
        """
        if not route:
            return 0
        
        # Distance from depot to first customer
        cost = dist_matrix[0][route[0]]  
        
        # Distance between consecutive customers in route
        for i in range(len(route) - 1):
            cost += dist_matrix[route[i]][route[i + 1]] 
        
        # Distance from last customer back to depot
        cost += dist_matrix[route[-1]][0]
        return cost
    
    def is_feasible(self):
        """Check if all routes satisfy capacity constraints.
        
        Returns:
            True if all routes are feasible, False otherwise
        """
        for route in self.routes:
            route_demand = sum(demands[customer] for customer in route)
            if route_demand > capacity:
                return False
        return True
    
    def print_solution(self):
        """Print solution details including cost and route assignments."""
        print(f"Total Cost: {self.total_cost()}")
        print(f"Number of vehicles used: {len(self.routes)}")
        
        for i, route in enumerate(self.routes):
            print(f"  Vehicle {i+1}: {route}")
        
        if self.unassigned:
            print(f"  Unassigned customers: {self.unassigned}")

def can_merge(route1, route2, customer_i, customer_j):
    """Check if two routes can be merged via connecting customers i and j.
    
    For Clarke-Wright, routes can only be merged if:
    1. Combined demand doesn't exceed vehicle capacity
    2. Both connecting customers are at route endpoints (first or last)
    
    Args:
        route1, route2: Routes to potentially merge
        customer_i, customer_j: Customers that would be connected
        
    Returns:
        True if routes can be merged, False otherwise
    """
    # Check capacity constraint
    demand1 = sum(demands[customer] for customer in route1)
    demand2 = sum(demands[customer] for customer in route2)
    
    if demand1 + demand2 > capacity:
        return False
    
    # Check if customers are at route endpoints (required for valid merge)
    i_at_end = (customer_i == route1[0] or customer_i == route1[-1])
    j_at_end = (customer_j == route2[0] or customer_j == route2[-1])
    
    return i_at_end and j_at_end

def merge_routes(route1, route2, customer_i, customer_j):
    """Merge two routes by connecting them via customers i and j.
    
    Handles all possible endpoint connection cases:
    - End of route1 to start of route2
    - End of route1 to end of route2 (reverse route2)
    - Start of route1 to start of route2 (reverse route1)
    - Start of route1 to end of route2
    
    Args:
        route1, route2: Routes to merge
        customer_i, customer_j: Connecting customers
        
    Returns:
        Merged route as a single list
    """
    if customer_i == route1[-1] and customer_j == route2[0]:
        return route1 + route2
    elif customer_i == route1[-1] and customer_j == route2[-1]:
        return route1 + route2[::-1]  # Reverse route2
    elif customer_i == route1[0] and customer_j == route2[0]:
        return route1[::-1] + route2  # Reverse route1
    elif customer_i == route1[0] and customer_j == route2[-1]:
        return route2 + route1
    else:
        # Fallback (shouldn't reach here if can_merge was called correctly)
        return route1 + route2

def solve_cw(customers):
    """Solve VRP using Clarke-Wright Savings Algorithm.

    1. Start with individual routes for each customer (depot-customer-depot)
    2. Calculate savings for merging each pair of routes
    3. Sort savings in descending order
    4. Merge routes with highest savings if feasible
    5. Continue until no more beneficial merges possible
    
    Savings formula: s_ij = d_0i + d_0j - d_ij
    (distance saved by connecting customers i,j directly vs via depot)
    
    Args:
        customers: List of customer IDs to route
        
    Returns:
        Solution object with routes and any unassigned customers
    """
    if not customers:
        return Solution([])
    
    # Step 1: Initialize with individual routes for each customer
    routes = []
    for customer in customers:
        routes.append([customer])

    # Step 2: Calculate all pairwise savings
    savings_list = []
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            customer_i = customers[i]
            customer_j = customers[j]
            
            # Clarke-Wright savings formula
            saving = (dist_matrix[0][customer_i] + 
                     dist_matrix[0][customer_j] - 
                     dist_matrix[customer_i][customer_j])
            
            savings_list.append((saving, customer_i, customer_j))

    # Step 3: Sort savings in descending order (highest savings first)
    savings_list.sort(reverse=True)
    
    # Step 4: Process savings and merge routes when beneficial and feasible
    for saving_value, customer_i, customer_j in savings_list:
        if saving_value <= 0:
            break  # No more beneficial merges possible
        
        # Find which routes contain customers i and j
        route_i_idx = None
        route_j_idx = None
        
        for idx, route in enumerate(routes):
            if customer_i in route:
                route_i_idx = idx
            if customer_j in route:
                route_j_idx = idx
        
        # Skip if customers are already in the same route
        if route_i_idx == route_j_idx:
            continue
        
        route_i = routes[route_i_idx]
        route_j = routes[route_j_idx]
        
        # Merge routes if feasible (capacity + endpoint constraints)
        if can_merge(route_i, route_j, customer_i, customer_j):
            merged_route = merge_routes(route_i, route_j, customer_i, customer_j)
            
            # Replace first route with merged route, remove second route
            routes[route_i_idx] = merged_route
            routes.pop(route_j_idx if route_j_idx > route_i_idx else route_j_idx)
    
    # Step 5: Separate feasible routes from infeasible ones
    # (This shouldn't happen with proper can_merge checking, but safety check)
    feasible_routes = []
    unassigned = []
    
    for route in routes:
        route_demand = sum(demands[customer] for customer in route)
        if route_demand <= capacity:
            feasible_routes.append(route)
        else:
            # If somehow infeasible, mark customers as unassigned
            unassigned.extend(route)
    
    return Solution(feasible_routes, unassigned)

def main():
    global coords, demands, capacity, vehicle_count, dist_matrix
    
    # Load problem instance
    folder = "../dataset/files"
    filename = input("Enter filename (e.g. c1_25.txt): ")
    
    coords, demands, capacity, vehicle_count = load_data(f"../../dataset/{folder}/{filename}")
    dist_matrix = calc_distances(coords)
    
    # Generate dynamic customer arrival scenario
    static_customers, dynamic_customers = Poisson(
        len(demands), STATIC_RATIO, LAMBDA_RATE, seed=SEED
    )
    
    start_time = time.time()
    
    # Solve initial problem with only static customers
    current_solution = solve_cw(list(static_customers))
    
    current_solution.print_solution()
    
    # Main simulation loop - process dynamic customer arrivals
    for t in range(1, MAX_SIM_TIME + 1):
        # Check for new customer arrivals at time t
        new_arrivals = [customer for customer, arrival_time in dynamic_customers.items() 
                       if arrival_time == t]
        
        if not new_arrivals:
            continue  # No arrivals, continue to next time step
        
        # Re-solve entire problem with all revealed customers
        all_customers = []
        for route in current_solution.routes:
            all_customers.extend(route)
        all_customers.extend(current_solution.unassigned)
        all_customers.extend(new_arrivals)
        
        current_solution = solve_cw(all_customers)
        
        # Check if all customers have arrived
        all_arrived = all(arrival_time <= t for arrival_time in dynamic_customers.values())
        if all_arrived:
            break  # All customers processed, can terminate early
    
    # Report final results
    solve_time = time.time() - start_time
    current_solution.print_solution()

if __name__ == "__main__":
    main()