import os
import sys
import time
import numpy as np
import pandas as pd
import numpy.random as rnd
import json
from io import StringIO
from contextlib import redirect_stdout

# Import Clarke-Wright
from CW import main as cw_main

# Configuration
NUM_RUNS = 20

def load_optimal_solutions():
    """Load optimal solutions from JSON file"""
    try:
        with open("optimal_solutions.json", "r") as f:
            data = json.load(f)
        
        # Parse the PyVRP results to extract cost and vehicle count
        optimal_solutions = {}
        for instance, result_data in data.items():
            if result_data['success']:
                result_text = result_data['result']
                
                # Parse both objective (total) and distance
                total_cost = None
                distance = None
                vehicles = None
                for line in result_text.split('\n'):
                    if 'objective:' in line:
                        total_cost = float(line.split(':')[1].strip())
                    elif 'distance:' in line:
                        distance = float(line.split(':')[1].strip())
                    elif '# routes:' in line:
                        vehicles = int(line.split(':')[1].strip())
                
                optimal_solutions[instance] = {
                    'optimal_total_cost': total_cost,
                    'optimal_distance': distance,
                    'vehicles_used': vehicles,
                    'success': True
                }
            else:
                optimal_solutions[instance] = {
                    'optimal_total_cost': None,
                    'optimal_distance': None,
                    'vehicles_used': None,
                    'success': False
                }
        
        return optimal_solutions
        
    except FileNotFoundError:
        print("Error: optimal_solutions.json not found. Run the optimal solver first.")
        return {}

def run_clarke_wright(instance, seed):
    """Run Clarke-Wright and capture metrics"""
    start_time = time.time()
    
    # Set seeds
    np.random.seed(seed)
    rnd.seed(seed)
    
    # Mock input for filename
    original_input = input
    sys.modules['builtins'].input = lambda _: instance
    
    try:
        # Capture stdout to get solution info
        f = StringIO()
        with redirect_stdout(f):
            cw_main()
        
        computation_time = time.time() - start_time
        output = f.getvalue()
        
        # Parse output for metrics
        cost = None
        vehicles = None
        routes = []
        feasible = True
        
        lines = output.split('\n')
        for line in lines:
            if "Total Cost:" in line:
                cost = float(line.split(':')[1].strip())
            elif "Number of vehicles used:" in line:
                vehicles = int(line.split(':')[1].strip())
            elif "  Vehicle " in line and ": [" in line:
                # Extract route from line like "  Vehicle 1: [1, 2, 3]"
                try:
                    route_part = line.split(': [')[1].rstrip(']')
                    if route_part.strip():
                        route = [int(x.strip()) for x in route_part.split(',') if x.strip()]
                        if route:
                            routes.append([0] + route + [0])  # Add depot start/end
                except:
                    pass
            elif "  Unassigned customers:" in line:
                # Check if there are unassigned customers
                unassigned_part = line.split(':')[1].strip()
                if unassigned_part != "[]" and unassigned_part:
                    feasible = False
        
        return {
            "algorithm": "Clarke-Wright",
            "instance": instance,
            "seed": seed,
            "solution_cost": cost if cost else float('inf'),
            "computation_time": computation_time,
            "num_vehicles": vehicles if vehicles else 0,
            "routes": routes,
            "feasible": feasible,
            "success": cost is not None,
            "error": None
        }
        
    except Exception as e:
        return {
            "algorithm": "Clarke-Wright",
            "instance": instance,
            "seed": seed,
            "solution_cost": float('inf'),
            "computation_time": time.time() - start_time,
            "num_vehicles": 0,
            "routes": [],
            "feasible": False,
            "success": False,
            "error": str(e)
        }
    finally:
        # Restore input
        sys.modules['builtins'].input = original_input

def evaluate_clarke_wright(optimal_solutions):
    """Evaluate Clarke-Wright across all instances"""
    print("Evaluating Clarke-Wright Algorithm")
    print(f"Running {NUM_RUNS} runs per instance")
    
    # Get instances from optimal solutions
    instances = list(optimal_solutions.keys())
    
    all_results = []
    best_routes = {}
    
    # Generate seeds
    np.random.seed(42)
    seeds = np.random.randint(1, 100000, NUM_RUNS).tolist()
    
    for instance in instances:
        print(f"\nProcessing {instance}...")
        
        instance_results = []
        best_cost = float('inf')
        best_route = None
        
        for run_idx, seed in enumerate(seeds):
            print(f"  Run {run_idx + 1}/{NUM_RUNS}", end=" ")
            
            result = run_clarke_wright(instance, seed)
            instance_results.append(result)
            
            # Track best route for this instance
            if result['success'] and result['solution_cost'] < best_cost:
                best_cost = result['solution_cost']
                best_route = result.get('routes', [])
            
            if result['success']:
                print(f"✓ Cost: {result['solution_cost']:.2f}")
            else:
                print(f"✗ Failed")
        
        # Store best route (ensure routes stay on same line)
        if best_route:
            best_routes[instance] = {
                "cost": best_cost,
                "routes": best_route
            }
        
        all_results.extend(instance_results)
    
    return all_results, best_routes

def calculate_metrics(results, optimal_solutions):
    """Calculate all performance metrics"""
    df = pd.DataFrame(results)
    metrics = []
    
    for instance in df['instance'].unique():
        instance_data = df[df['instance'] == instance]
        successful_runs = instance_data[instance_data['success'] == True]
        
        # Get optimal cost and distance
        optimal_total_cost = None
        optimal_distance = None
        if instance in optimal_solutions and optimal_solutions[instance]['success']:
            optimal_total_cost = optimal_solutions[instance]['optimal_total_cost']
            optimal_distance = optimal_solutions[instance]['optimal_distance']
        
        if len(successful_runs) > 0:
            costs = successful_runs['solution_cost']
            times = successful_runs['computation_time']
            vehicles = successful_runs['num_vehicles']
            feasible_runs = successful_runs[successful_runs['feasible'] == True]
            
            # Basic metrics
            success_rate = len(successful_runs) / len(instance_data)
            feasibility_rate = len(feasible_runs) / len(instance_data)
            best_cost = costs.min()
            avg_cost = costs.mean()
            worst_cost = costs.max()
            cost_std = costs.std()
            avg_time = times.mean()
            time_std = times.std()
            avg_vehicles = vehicles.mean()
            vehicles_std = vehicles.std()
            
            # Optimality metrics - use distance for comparison since CW only optimizes distance
            deviation_from_optimal = None
            within_95_percent = 0.0
            within_99_percent = 0.0
            
            # Calculate distance-only cost for fair comparison
            distance_costs = costs - (vehicles * 50)  # Remove vehicle costs to get distance only
            
            if optimal_distance:
                best_distance = distance_costs.min()
                deviation_from_optimal = ((best_distance - optimal_distance) / optimal_distance) * 100
                within_95_percent = sum(1 for c in distance_costs if c <= optimal_distance * 1.05) / len(distance_costs)
                within_99_percent = sum(1 for c in distance_costs if c <= optimal_distance * 1.01) / len(distance_costs)
            
            # Robustness (coefficient of variation)
            robustness_cost = cost_std / avg_cost if avg_cost > 0 else 0
            robustness_time = time_std / avg_time if avg_time > 0 else 0
            
        else:
            # All runs failed
            success_rate = 0.0
            feasibility_rate = 0.0
            best_cost = float('inf')
            avg_cost = float('inf')
            worst_cost = float('inf')
            cost_std = 0.0
            avg_time = instance_data['computation_time'].mean()
            time_std = instance_data['computation_time'].std()
            avg_vehicles = 0.0
            vehicles_std = 0.0
            deviation_from_optimal = float('inf') if optimal_distance else None
            within_95_percent = 0.0
            within_99_percent = 0.0
            robustness_cost = 0.0
            robustness_time = 0.0
        
        metrics.append({
            "algorithm": "Clarke-Wright",
            "instance": instance,
            "instance_type": instance[0].upper(),
            "instance_size": int(instance.split('_')[1].split('.')[0]),
            
            # Solution Quality
            "optimal_total_cost": optimal_total_cost,
            "optimal_distance": optimal_distance,
            "best_cost": best_cost,
            "avg_cost": avg_cost,
            "worst_cost": worst_cost,
            "cost_std": cost_std,
            "deviation_from_optimal_percent": deviation_from_optimal,
            
            # Computational Time
            "avg_computation_time": avg_time,
            "computation_time_std": time_std,
            
            # Feasibility
            "success_rate": success_rate,
            "feasibility_rate": feasibility_rate,
            
            # Robustness
            "cost_robustness_cv": robustness_cost,
            "time_robustness_cv": robustness_time,
            
            # Success Rate
            "within_95_percent_optimal": within_95_percent,
            "within_99_percent_optimal": within_99_percent,
            
            # Vehicle Usage
            "avg_vehicles": avg_vehicles,
            "vehicles_std": vehicles_std
        })
    
    return pd.DataFrame(metrics)

def main():
    """Main evaluation function"""
    # Load optimal solutions
    optimal_solutions = load_optimal_solutions()
    if not optimal_solutions:
        print("Cannot proceed without optimal solutions.")
        return None, None, None
    
    # Run evaluation
    results, best_routes = evaluate_clarke_wright(optimal_solutions)
    
    # Calculate metrics
    metrics_df = calculate_metrics(results, optimal_solutions)
    
    # Save detailed results
    pd.DataFrame(results).to_csv("clarke_wright_detailed_results.csv", index=False)
    
    # Save metrics
    metrics_df.to_csv("clarke_wright_metrics.csv", index=False)
    
    # Save best routes (compact JSON format)
    with open("clarke_wright_best_routes.json", "w") as f:
        json.dump(best_routes, f, separators=(',', ':'))
    
    print(f"\nResults saved:")
    print(f"  Detailed results: clarke_wright_detailed_results.csv")
    print(f"  Metrics: clarke_wright_metrics.csv")
    print(f"  Best routes: clarke_wright_best_routes.json")
    
    # Print summary
    successful_instances = metrics_df[metrics_df['success_rate'] > 0]
    if len(successful_instances) > 0:
        print(f"\nSummary ({len(successful_instances)} successful instances):")
        print(f"  Average Success Rate: {successful_instances['success_rate'].mean():.3f}")
        print(f"  Average Feasibility Rate: {successful_instances['feasibility_rate'].mean():.3f}")
        print(f"  Average Best Cost: {successful_instances['best_cost'].mean():.2f}")
        print(f"  Average Computation Time: {successful_instances['avg_computation_time'].mean():.3f}s")
        
        # Deviation from optimal
        valid_deviations = successful_instances['deviation_from_optimal_percent'].dropna()
        if len(valid_deviations) > 0:
            print(f"  Average Deviation from Optimal: {valid_deviations.mean():.2f}%")
            print(f"  Average Within 95% of Optimal: {successful_instances['within_95_percent_optimal'].mean():.3f}")
            print(f"  Average Within 99% of Optimal: {successful_instances['within_99_percent_optimal'].mean():.3f}")
        
        # Robustness
        print(f"  Average Cost Robustness (CV): {successful_instances['cost_robustness_cv'].mean():.3f}")
        print(f"  Average Time Robustness (CV): {successful_instances['time_robustness_cv'].mean():.3f}")
    
    return results, metrics_df, best_routes 

if __name__ == "__main__":
    results, metrics, best_routes = main()
