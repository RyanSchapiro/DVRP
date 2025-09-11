import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import time

from util import load_data, Poisson

def compute_distance_matrix(coords):
    distances = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=-1)
    return distances.astype(int)

SEED = 89
STATIC_RATIO = 0.7
LAMBDA_RATE = 0.4
MAX_SIMULATION_TIME = 1000
UNASSIGNED_PENALTY = 1000

rnd.seed(SEED)

class CVRPSolution:
    
    def __init__(self, routes, unassigned=None):
        self.routes = routes
        self.unassigned = unassigned if unassigned is not None else []
    
    def calculate_total_cost(self):
        total_distance = 0
        for route in self.routes:
            total_distance += self._calculate_route_cost(route)

        vehicle_cost = len(self.routes) * 50  # Add this line
        penalty_cost = len(self.unassigned) * UNASSIGNED_PENALTY
        return total_distance + penalty_cost + vehicle_cost
    
    def _calculate_route_cost(self, route):
        if not route:
            return 0
        
        cost = distance_matrix[0][route[0]]  
        for i in range(len(route) - 1):
            cost += distance_matrix[route[i]][route[i + 1]] 
        cost += distance_matrix[route[-1]][0]
        return cost
    
    def is_feasible(self):
        for route in self.routes:
            route_demand = sum(demands[customer] for customer in route)
            if route_demand > capacity:
                return False
        return True
    
    def print_solution(self):
        print(f"\nSolution Summary:")
        print(f"Total Cost: {self.calculate_total_cost()}")
        print(f"Number of vehicles used: {len(self.routes)}")
        
        total_distance = 0
        for i, route in enumerate(self.routes):
            route_cost = self._calculate_route_cost(route)
            route_demand = sum(demands[customer] for customer in route)
            total_distance += route_cost
            print(f"  Vehicle {i+1}: {route}")
            print(f"    Distance: {route_cost}, Load: {route_demand}/{capacity}")
        
        if self.unassigned:
            print(f"  Unassigned customers: {self.unassigned}")
        
        print(f"Total distance (excluding penalties): {total_distance}")

def solve_cvrp_clarke_wright(customers):

    if not customers:
        return CVRPSolution([])
    
    print(f"Solving CVRP using Clarke-Wright Savings for {len(customers)} customers...")
    
    routes = []
    for customer in customers:
        routes.append([customer])

    savings_list = []
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            customer_i = customers[i]
            customer_j = customers[j]
            

            saving = (distance_matrix[0][customer_i] + 
                     distance_matrix[0][customer_j] - 
                     distance_matrix[customer_i][customer_j])
            
            savings_list.append((saving, customer_i, customer_j))
    

    savings_list.sort(reverse=True)
    
    print(f"  Generated {len(savings_list)} savings pairs")
    if savings_list:
        print(f"  Best saving: {savings_list[0][0]} for customers {savings_list[0][1]}-{savings_list[0][2]}")
    

    merges_performed = 0
    
    for saving_value, customer_i, customer_j in savings_list:
        if saving_value <= 0:
            break 
        
        route_i_idx = None
        route_j_idx = None
        
        for idx, route in enumerate(routes):
            if customer_i in route:
                route_i_idx = idx
            if customer_j in route:
                route_j_idx = idx
        
    
        if route_i_idx == route_j_idx:
            continue
        
        route_i = routes[route_i_idx]
        route_j = routes[route_j_idx]
        

        if can_merge_routes(route_i, route_j, customer_i, customer_j):
    
            merged_route = merge_routes(route_i, route_j, customer_i, customer_j)
            
            routes[route_i_idx] = merged_route
            routes.pop(route_j_idx if route_j_idx > route_i_idx else route_j_idx)
            
            merges_performed += 1
    
    print(f"  Performed {merges_performed} route merges")
    print(f"  Final number of routes: {len(routes)}")
    
    feasible_routes = []
    unassigned = []
    
    for route in routes:
        route_demand = sum(demands[customer] for customer in route)
        if route_demand <= capacity:
            feasible_routes.append(route)
        else:

            unassigned.extend(route)
            print(f"  Warning: Route {route} exceeds capacity ({route_demand}/{capacity})")
    
    return CVRPSolution(feasible_routes, unassigned)

def can_merge_routes(route1, route2, customer_i, customer_j):

    demand1 = sum(demands[customer] for customer in route1)
    demand2 = sum(demands[customer] for customer in route2)
    
    if demand1 + demand2 > capacity:
        return False
    
    i_at_end = (customer_i == route1[0] or customer_i == route1[-1])
    j_at_end = (customer_j == route2[0] or customer_j == route2[-1])
    
    return i_at_end and j_at_end

def merge_routes(route1, route2, customer_i, customer_j):
   

    if customer_i == route1[-1] and customer_j == route2[0]:
        return route1 + route2
    elif customer_i == route1[-1] and customer_j == route2[-1]:
        return route1 + route2[::-1]
    elif customer_i == route1[0] and customer_j == route2[0]:
        return route1[::-1] + route2
    elif customer_i == route1[0] and customer_j == route2[-1]:
        return route2 + route1
    else:
        print(f"Warning: Unexpected merge case for {customer_i} in {route1} and {customer_j} in {route2}")
        return route1 + route2

def main():
    global coords, demands, capacity, vehicle_count, distance_matrix
    
    folder = "../dataset/files"
    filename = input("Enter filename (e.g. c1_25.txt): ")
    
    coords, demands, capacity, vehicle_count = load_data(f"../../dataset/{folder}/{filename}")
    distance_matrix = compute_distance_matrix(coords)
    
    print(f"Problem loaded:")
    print(f"  Customers: {len(demands) - 1}")
    print(f"  Vehicle capacity: {capacity}")
    print(f"  Vehicle count: {vehicle_count}")
    print(f"  Total demand: {sum(demands[1:])}")
    
    static_customers, dynamic_customers = Poisson(
        len(demands), STATIC_RATIO, LAMBDA_RATE, seed=SEED
    )
    
    print(f"\nCustomer distribution:")
    print(f"  Static customers: {len(static_customers)}")
    print(f"  Dynamic customers: {len(dynamic_customers)}")
    
    print(f"\n{'='*50}")
    print("SOLVING INITIAL PROBLEM (STATIC CUSTOMERS)")
    print(f"{'='*50}")
    
    start_time = time.time()
    current_solution = solve_cvrp_clarke_wright(list(static_customers))
    
    print(f"Initial solution found")
    current_solution.print_solution()

    print(f"\n{'='*50}")
    print("DYNAMIC CUSTOMER SIMULATION")
    print(f"{'='*50}")
    
    for t in range(1, MAX_SIMULATION_TIME + 1):
        new_arrivals = [customer for customer, arrival_time in dynamic_customers.items() 
                       if arrival_time == t]
        
        if not new_arrivals:
            continue
        
        print(f"\nTime {t}: New customers arrived: {new_arrivals}")
        
        all_customers = []
        for route in current_solution.routes:
            all_customers.extend(route)
        all_customers.extend(current_solution.unassigned)
        all_customers.extend(new_arrivals)
        
        print(f"Re-optimizing with {len(all_customers)} total customers...")
        start_time = time.time()
        current_solution = solve_cvrp_clarke_wright(all_customers)
        solve_time = time.time() - start_time
        print(f"Re-optimization completed in {solve_time:.3f} seconds")
        
        all_arrived = all(arrival_time <= t for arrival_time in dynamic_customers.values())
        if all_arrived:
            print(f"\nAll dynamic customers have arrived by time {t}")
            break
    
    solve_time = time.time() - start_time
    print(f"\n{'='*50}")
    print("FINAL SOLUTION")
    print(f"{'='*50}")
    print(f"Solved in {solve_time:.3f} seconds")
    print(f"Final solution:")
    current_solution.print_solution()
    
    if current_solution.routes:
        avg_route_length = np.mean([len(route) for route in current_solution.routes])
        avg_route_cost = np.mean([current_solution._calculate_route_cost(route) for route in current_solution.routes])
        avg_route_utilization = np.mean([sum(demands[c] for c in route) / capacity for route in current_solution.routes])
        
        print(f"\nSolution Statistics:")
        print(f"  Average route length: {avg_route_length:.1f} customers")
        print(f"  Average route cost: {avg_route_cost:.1f}")
        print(f"  Average capacity utilization: {avg_route_utilization*100:.1f}%")
        print(f"  Vehicle utilization: {len(current_solution.routes)}/{vehicle_count}")
    
    #plot_solution(current_solution, "Final CVRP Solution (Clarke-Wright)")

if __name__ == "__main__":
    main()