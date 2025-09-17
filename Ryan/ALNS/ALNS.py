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

SEED = 89                
STATIC_RATIO = 0.7         
LAMBDA_RATE = 0.4       
MAX_SIM_TIME = 1000         # Maximum simulation time steps
REOPT_ITERS = 3000          # ALNS iterations per reoptimization
PENALTY = 1000              # Penalty cost for unassigned customers
DESTROY_RATE = 0.2          # Fraction of customers to remove in destroy operators
MAX_STRINGS = 2             # Maximum number of route strings to destroy
MAX_STRING_LEN = 6          # Maximum length of string removal

rnd.seed(SEED)

def calc_distances(coords):
    """Calculate Euclidean distance matrix between all coordinate pairs.
    
    Args:
        coords: numpy array of (x,y) coordinates for depot and customers
        
    Returns:
        Integer distance matrix where distances[i][j] = distance from i to j
    """
    distances = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=-1)
    return distances.astype(int)

class State:
    """VRP Solution State"""
    
    def __init__(self, routes, unassigned=None):
        """Initialize state with routes and unassigned customers.
        
        Args:
            routes: List of routes, where each route is a list of customer IDs
            unassigned: List of customer IDs not yet assigned to routes
        """
        self.routes = routes
        self.unassigned = unassigned or []

    def copy(self):
        """Create deep copy of the state."""
        return State(copy.deepcopy(self.routes), self.unassigned.copy())

    def objective(self):
        """Calculate total solution cost including routing, vehicle, and penalty costs.
        
        Returns:
            Total cost = route costs + vehicle fixed costs + unassigned penalties
        """
        # Sum travel costs for all routes
        total_cost = sum(self._route_cost(route) for route in self.routes)
        
        # Fixed cost per vehicle
        vehicle_cost = len(self.routes) * 50 
        
        # Add penalty for unassigned customers
        total_cost += vehicle_cost + len(self.unassigned) * PENALTY
        return total_cost

    @property
    def cost(self):
        """Alias for objective() method."""
        return self.objective()

    def find_route(self, customer):
        """Find which route contains the given customer.
        
        Args:
            customer: Customer ID to search for
            
        Returns:
            The route (list) containing the customer
            
        Raises:
            ValueError: If customer not found in any route
        """

        for route in self.routes:
            if customer in route:
                return route
        raise ValueError(f"Customer {customer} not found in any route")

    def _route_cost(self, route):
        """Calculate total travel cost for a single route.
        
        Args:
            route: List of customer IDs in visit order
            
        Returns:
            Total distance starting and ending at depot (customer 0)
        """
        tour = [0] + route + [0]  # Add depot at start and end
        return sum(dist_matrix[tour[i]][tour[i+1]] for i in range(len(tour) - 1))

def remove_empty(state):
    """Remove empty routes from state (cleanup utility function)."""
    state.routes = [route for route in state.routes if route]
    return state

def get_neighbors(customer):
    """Get customers sorted by distance from given customer.
    
    Args:
        customer: Reference customer ID
        
    Returns:
        List of customer IDs sorted by increasing distance (excluding depot)
    """
    distances = np.argsort(dist_matrix[customer])
    return [i for i in distances if i != 0]  # Exclude depot (customer 0)

def remove_string_section(route, customer, max_size, rng):
    """Remove a contiguous string of customers from route centered around given customer.
    
    Args:
        route: Route to remove customers from
        customer: Central customer around which to remove string
        max_size: Maximum number of customers to remove
        rng: Random number generator
        
    Returns:
        List of removed customers
    """
    # Determine actual removal size (1 to max_size, limited by route length)
    size = rng.integers(1, min(len(route), max_size) + 1)
    
    # Choose starting position
    start = route.index(customer) - rng.integers(size)
    indices = [idx % len(route) for idx in range(start, start + size)]
    
    # Remove customers in reverse order to maintain indices
    removed = []
    for idx in sorted(indices, reverse=True):
        removed.append(route.pop(idx))
    return removed

def random_destroy(state, rng):
    """Destroy operator: randomly remove a fraction of assigned customers.
    
    Args:
        state: Current solution state
        rng: Random number generator
        
    Returns:
        New state with customers removed and added to unassigned list
    """
    destroyed = state.copy()
    assigned = [c for route in destroyed.routes for c in route]
    
    if not assigned:
        return destroyed
    
    # Remove DESTROY_RATE fraction of customers (at least 1)
    num_remove = max(1, int(len(assigned) * DESTROY_RATE))
    num_remove = min(num_remove, len(assigned))
    
    # Randomly select customers to remove
    for customer in rng.choice(assigned, num_remove, replace=False).tolist():
        destroyed.unassigned.append(customer)
        route = destroyed.find_route(customer)
        route.remove(customer)
    
    return remove_empty(destroyed)

def string_destroy(state, rng):
    """Destroy operator: remove contiguous strings of customers from routes.
    
    Args:
        state: Current solution state
        rng: Random number generator
        
    Returns:
        New state with customer strings removed
    """
    destroyed = state.copy()
    if not state.routes:
        return destroyed
    
    # Determine string removal parameters
    avg_size = int(np.mean([len(route) for route in state.routes]))
    max_size = max(MAX_STRING_LEN, avg_size)
    max_removals = min(len(state.routes), MAX_STRINGS)
    destroyed_routes = []
    
    if num_customers > 1:
        # Pick random center customer and remove strings around nearby customers
        center = rng.integers(1, num_customers + 1)
        for customer in get_neighbors(center):
            if len(destroyed_routes) >= max_removals:
                break
            if customer in destroyed.unassigned:
                continue
            
            try:
                route = destroyed.find_route(customer)
                if route in destroyed_routes:  # Don't destroy same route twice
                    continue
                
                # Remove string section around this customer
                removed = remove_string_section(route, customer, max_size, rng)
                destroyed.unassigned.extend(removed)
                destroyed_routes.append(route)
            except ValueError:
                continue
    
    return destroyed

def can_insert(customer, route):
    """Check if customer can be inserted into route without violating capacity.
    
    Args:
        customer: Customer ID to insert
        route: Route to insert into
        
    Returns:
        True if insertion is feasible, False otherwise
    """
    total_demand = sum(demands[c] for c in route) + demands[customer]
    return total_demand <= capacity

def insertion_cost(customer, route, idx):
    """Calculate cost from inserting customer at given position in route.
    
    Args:
        customer: Customer to insert
        route: Route to insert into  
        idx: Position in route to insert (0 = start, len(route) = end)
        
    Returns:
        Change in route cost (can be negative if insertion shortens route)
    """
    pred = 0 if idx == 0 else route[idx - 1]        # Predecessor (depot if first)
    succ = 0 if idx == len(route) else route[idx]   # Successor (depot if last)
    
    # Cost = distance to predecessor + distance to successor - old direct distance
    return (dist_matrix[pred][customer] + 
            dist_matrix[customer][succ] - 
            dist_matrix[pred][succ])

def find_best_insert(customer, state):
    """Find best position to insert customer across all routes.
    
    Args:
        customer: Customer to insert
        state: Current state with routes
        
    Returns:
        Tuple of (best_route, best_position) or (None, None) if no valid insertion
    """
    best_cost, best_route, best_idx = None, None, None
    
    # Try inserting in every feasible position across all routes
    for route in state.routes:
        if not can_insert(customer, route):
            continue
        
        for idx in range(len(route) + 1):
            cost = insertion_cost(customer, route, idx)
            if best_cost is None or cost < best_cost:
                best_cost, best_route, best_idx = cost, route, idx
    
    return best_route, best_idx

def greedy_insert(state, rng):
    """Repair operator: greedily insert all unassigned customers.
    
    Args:
        state: State with unassigned customers
        rng: Random number generator
        
    Returns:
        Repaired state with all customers assigned
    """
    repaired = state.copy()
    rng.shuffle(repaired.unassigned)  # Random order for tie-breaking
    
    while repaired.unassigned:
        customer = repaired.unassigned.pop()
        route, idx = find_best_insert(customer, repaired)
        
        if route is not None:
            # Insert at best position in existing route
            route.insert(idx, customer)
        else:
            # Create new route if no feasible insertion
            repaired.routes.append([customer])
    
    return repaired

def main():
    global coords, demands, capacity, num_customers, dist_matrix
    
    # Load problem instance
    folder = "../../dataset/files"
    filename = input("Enter filename (e.g. c1_25.txt): ")
    
    coords, demands, capacity, vehicle_count = load_data(f"{folder}/{filename}")
    num_customers = len(demands) - 1  # Exclude depot
    dist_matrix = calc_distances(coords)
    
    # Generate dynamic customer arrival scenario
    static_customers, dynamic_customers = Poisson(
        len(demands), STATIC_RATIO, LAMBDA_RATE, seed=SEED
    )
    
    start_time = time.time()
    
    # Initialize ALNS algorithm
    alns = ALNS(rnd)
    alns.add_destroy_operator(random_destroy)   # Add destroy operators
    alns.add_destroy_operator(string_destroy)
    alns.add_repair_operator(greedy_insert)     # Add repair operator
    
    # Configure operator selection (Roulette Wheel with weights [25,5,1,0])
    select = RouletteWheel([25, 5, 1, 0], 0.8, 1, 1)
    
    # Create initial solution with only static customers
    initial_state = State(routes=[], unassigned=list(static_customers))
    current_solution = greedy_insert(initial_state, rnd)
    
    # Main simulation loop - process dynamic customer arrivals
    for t in range(1, MAX_SIM_TIME + 1):
        # Check for new customer arrivals at time t
        new_customers = [c for c, arrival_time in dynamic_customers.items() 
                        if arrival_time == t]
        
        if not new_customers:
            continue  # No arrivals, continue to next time step
        
        # Add new customers to unassigned list
        current_solution.unassigned.extend(new_customers)
        
        # Reoptimize solution using ALNS
        # Configure acceptance criterion and stopping condition
        accept = RecordToRecordTravel.autofit(
            current_solution.objective(), 0.02, 0, REOPT_ITERS
        )
        stop = MaxIterations(REOPT_ITERS)
        
        # Run ALNS optimization
        result = alns.iterate(current_solution, select, accept, stop)
        current_solution = result.best_state
        
        # Check if all customers have arrived and been assigned
        all_revealed = all(arrival_time <= t for c, arrival_time in dynamic_customers.items())
        if all_revealed and not current_solution.unassigned:
            break  # Problem solved
    
    # Report final results
    final_time = time.time() - start_time   
    objective = current_solution.objective()
    
    print(f"Final cost: {objective}")
    print(f"Vehicles used: {len(current_solution.routes)}")
    
    for i, route in enumerate(current_solution.routes):
        print(f"  Vehicle {i+1}: {route}")
    
    if current_solution.unassigned:
        print(f"Unassigned: {current_solution.unassigned}")

if __name__ == "__main__":
    main()