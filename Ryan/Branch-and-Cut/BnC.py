import sys
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import time

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("ERROR: Gurobi not available.")
    sys.exit(1)

from util import load_data, Poisson

def compute_distance_matrix(coords):
    n = len(coords)
    distance_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = int(np.sqrt((coords[i][0] - coords[j][0])**2 + 
                                                  (coords[i][1] - coords[j][1])**2))
    return distance_matrix

# Configuration
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
        
        penalty_cost = len(self.unassigned) * UNASSIGNED_PENALTY
        return total_distance + penalty_cost
    
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

def solve_cvrp_bnc(customers):
   
    if not customers:
        return CVRPSolution([])
    
    print(f"Solving CVRP using Branch-and-Cut for {len(customers)} customers...")
    
    # Create Gurobi model
    model = gp.Model("CVRP_BnC")
    model.setParam('OutputFlag', 1)  # Show Gurobi progress
    model.setParam('TimeLimit', 60)  # Reduce to 1 minute for testing
    model.setParam('MIPGap', 0.10)   # Accept 10% gap for faster solutions
    model.setParam('LogToConsole', 1)  # Enable console logging
    model.setParam('Heuristics', 0.5)  # Spend more time on heuristics
    model.setParam('Cuts', 2)  # Aggressive cutting planes
    
    nodes = [0] + customers
    n = len(nodes)
    
    # Decision variables
    print("Creating decision variables...")
    # x[i,j] = 1 if vehicle travels from node i to node j
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[i,j] = model.addVar(vtype=GRB.BINARY, name=f'x_{nodes[i]}_{nodes[j]}')
    
    # u[i] = cumulative load when leaving node i (MTZ variables)
    u = {}
    for i in range(n):
        if i == 0:  # depot has no load
            u[i] = model.addVar(lb=0, ub=0, name=f'u_{nodes[i]}')
        else:  # customers
            u[i] = model.addVar(lb=0, ub=capacity, name=f'u_{nodes[i]}')
    
    print(f"Created {len([x for x in x.values()])} binary variables and {len(u)} continuous variables")
    
    # Objective: minimize total travel distance
    print("Setting up objective function...")
    obj = gp.quicksum(distance_matrix[nodes[i]][nodes[j]] * x[i,j] 
                      for i in range(n) for j in range(n) if i != j)
    model.setObjective(obj, GRB.MINIMIZE)
    
    
    print("Adding constraints...")
    
    for j in range(1, n):
        model.addConstr(
            gp.quicksum(x[i,j] for i in range(n) if i != j) == 1,
            name=f'visit_customer_{nodes[j]}'
        )
    
    for i in range(1, n):
        model.addConstr(
            gp.quicksum(x[i,j] for j in range(n) if i != j) == 
            gp.quicksum(x[j,i] for j in range(n) if j != i),
            name=f'flow_conservation_{nodes[i]}'
        )
    
    model.addConstr(
        gp.quicksum(x[0,j] for j in range(1, n)) <= vehicle_count,
        name='vehicle_limit'
    )
    
    print("Adding MTZ subtour elimination constraints...")
    # 4. MTZ subtour elimination and capacity constraints
    M = capacity + max(demands) if len(demands) > 0 else capacity + 100
    constraint_count = 0
    for i in range(n):
        for j in range(1, n):  # only for customer nodes
            if i != j:
                model.addConstr(
                    u[j] >= u[i] + demands[nodes[j]] - M * (1 - x[i,j]),
                    name=f'mtz_{nodes[i]}_{nodes[j]}'
                )
                constraint_count += 1
    
    print(f"Added {constraint_count} MTZ constraints")
    
    # 5. Load constraints for customers
    for i in range(1, n):
        model.addConstr(u[i] >= demands[nodes[i]], name=f'min_load_{nodes[i]}')
        model.addConstr(u[i] <= capacity, name=f'max_load_{nodes[i]}')
    
    print(f"Model setup complete: {model.numVars} variables, {model.numConstrs} constraints")
    # Solve the model
    print("\n" + "="*50)
    print("STARTING GUROBI OPTIMIZATION")
    print("="*50)
    print("Progress will be shown below...")
    
    start_time = time.time()
    model.optimize()
    solve_time = time.time() - start_time
    
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE")
    print("="*50)
    
    # Extract solution
    if model.status == GRB.OPTIMAL:
        print(f"✓ OPTIMAL solution found in {solve_time:.2f}s")
        print(f"  Objective value: {model.objVal}")
        print(f"  Gap: {model.MIPGap*100:.2f}%")
        return extract_routes_from_solution(model, x, nodes, customers)
    elif model.status == GRB.TIME_LIMIT and model.SolCount > 0:
        print(f"⚠ TIME LIMIT reached, best solution found in {solve_time:.2f}s")
        print(f"  Best objective value: {model.objVal}")
        print(f"  Gap: {model.MIPGap*100:.2f}%")
        return extract_routes_from_solution(model, x, nodes, customers)
    elif model.status == GRB.INFEASIBLE:
        print(f"✗ Problem is INFEASIBLE")
        return CVRPSolution([], customers)
    else:
        print(f"✗ No feasible solution found (status: {model.status})")
        return CVRPSolution([], customers)

def extract_routes_from_solution(model, x_vars, nodes, customers):
    """Extract routes from the Gurobi solution."""
    print("Extracting routes from solution...")
    
    # Build adjacency list from solution
    outgoing = {node: [] for node in nodes}
    
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j and x_vars[i,j].x > 0.5:  # Edge is used
                outgoing[nodes[i]].append(nodes[j])
    
    print(f"Found {sum(len(adj) for adj in outgoing.values())} active edges")
    
    # Extract routes by following paths from depot
    routes = []
    visited = set()
    
    # Start from depot and follow each outgoing edge
    for next_node in outgoing[0]:
        if next_node in visited:
            continue
        
        route = []
        current = next_node
        
        # Follow the path until returning to depot
        while current != 0 and current not in visited:
            route.append(current)
            visited.add(current)
            
            # Find next node
            if current in outgoing and len(outgoing[current]) > 0:
                current = outgoing[current][0]
            else:
                break
        
        if route:
            routes.append(route)
            print(f"  Extracted route: {route}")
    
    # Check for unvisited customers
    unassigned = [c for c in customers if c not in visited]
    if unassigned:
        print(f"  Warning: Unvisited customers: {unassigned}")
    
    return CVRPSolution(routes, unassigned)

def plot_solution(solution, title="CVRP Solution"):
    """Plot the CVRP solution."""
    plt.figure(figsize=(12, 8))
    
    # Plot depot
    plt.plot(coords[0, 0], coords[0, 1], 'rs', markersize=15, label='Depot')
    
    # Plot all customers
    for i in range(1, len(coords)):
        plt.plot(coords[i, 0], coords[i, 1], 'ko', markersize=8, alpha=0.6)
        plt.annotate(str(i), (coords[i, 0], coords[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Plot routes with different colors
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(solution.routes), 1)))
    
    for idx, route in enumerate(solution.routes):
        if not route:
            continue
        
        # Create tour: depot -> customers -> depot
        tour_coords = [coords[0]]  # start at depot
        for customer in route:
            tour_coords.append(coords[customer])
        tour_coords.append(coords[0])  # return to depot
        
        tour_coords = np.array(tour_coords)
        plt.plot(tour_coords[:, 0], tour_coords[:, 1], 'o-', 
                color=colors[idx], linewidth=2, markersize=6,
                label=f'Vehicle {idx+1}')
    
    # Highlight unassigned customers
    if solution.unassigned:
        for customer in solution.unassigned:
            plt.plot(coords[customer, 0], coords[customer, 1], 'rx', 
                    markersize=12, markeredgewidth=3, label='Unassigned' if customer == solution.unassigned[0] else "")
    
    plt.title(f'{title} (Cost: {solution.calculate_total_cost()}, Vehicles: {len(solution.routes)})')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def main():
    global coords, demands, capacity, vehicle_count, distance_matrix
    
    # Load problem instance
    folder = input("Enter folder name (inside ../dataset/): ")
    filename = input("Enter filename (e.g. c1_25.txt): ")
    
    coords, demands, capacity, vehicle_count = load_data(f"../dataset/{folder}/{filename}")
    distance_matrix = compute_distance_matrix(coords)
    
    print(f"Problem loaded:")
    print(f"  Customers: {len(demands) - 1}")
    print(f"  Vehicle capacity: {capacity}")
    print(f"  Vehicle count: {vehicle_count}")
    print(f"  Total demand: {sum(demands[1:])}")
    
    # Generate static and dynamic customers
    static_customers, dynamic_customers = Poisson(
        len(demands), STATIC_RATIO, LAMBDA_RATE, seed=SEED
    )
    
    print(f"\nCustomer distribution:")
    print(f"  Static customers: {len(static_customers)}")
    print(f"  Dynamic customers: {len(dynamic_customers)}")
    
    # Solve initial problem with static customers
    print(f"\n{'='*50}")
    print("SOLVING INITIAL PROBLEM (STATIC CUSTOMERS)")
    print(f"{'='*50}")
    
    current_solution = solve_cvrp_bnc(list(static_customers))
    current_solution.print_solution()
    
    # Simulate dynamic arrivals
    print(f"\n{'='*50}")
    print("DYNAMIC CUSTOMER SIMULATION")
    print(f"{'='*50}")
    
    for t in range(1, MAX_SIMULATION_TIME + 1):
        # Check for new customer arrivals
        new_arrivals = [customer for customer, arrival_time in dynamic_customers.items() 
                       if arrival_time == t]
        
        if not new_arrivals:
            continue
        
        print(f"\nTime {t}: New customers arrived: {new_arrivals}")
        
        # Collect all customers that need to be served
        all_customers = []
        for route in current_solution.routes:
            all_customers.extend(route)
        all_customers.extend(current_solution.unassigned)
        all_customers.extend(new_arrivals)
        
        # Re-solve with all customers
        print(f"Re-optimizing with {len(all_customers)} total customers...")
        current_solution = solve_cvrp_bnc(all_customers)
        
        # Check if all dynamic customers have arrived
        all_arrived = all(arrival_time <= t for arrival_time in dynamic_customers.values())
        if all_arrived:
            print(f"\nAll dynamic customers have arrived by time {t}")
            break
    
    # Final results
    print(f"\n{'='*50}")
    print("FINAL SOLUTION")
    print(f"{'='*50}")
    current_solution.print_solution()
    
    # Visualize solution
    plot_solution(current_solution, "Final CVRP Solution (Branch-and-Cut)")

if __name__ == "__main__":
    main()