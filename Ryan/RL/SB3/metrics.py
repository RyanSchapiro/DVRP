import os
import sys
import time
import numpy as np
import pandas as pd
import json
from stable_baselines3 import SAC

# Configuration
NUM_RUNS = 20
# Configuration
NUM_RUNS = 20

def get_model_path(instance):
    """Get the appropriate model path based on instance type"""
    instance_type = instance[0].upper()
    
    if instance_type == 'C':
        return "c50_200"  # Remove .zip
    elif instance_type == 'R':
        return "r50_200"  # Remove .zip
    elif instance.startswith('rc'):
        return "rc50_200"  # Remove .zip
    else:
        raise ValueError(f"Unknown instance type: {instance_type}")

def load_optimal_solutions():
    """Load optimal solutions from JSON file"""
    try:
        with open("../../optimal_solutions.json", "r") as f:
            data = json.load(f)
        
        optimal_solutions = {}
        for instance, result_data in data.items():
            if result_data['success']:
                result_text = result_data['result']
                
                # Parse objective cost for comparison
                objective = None
                vehicles = None
                for line in result_text.split('\n'):
                    if 'objective:' in line:
                        objective = float(line.split(':')[1].strip())
                    elif '# routes:' in line:
                        vehicles = int(line.split(':')[1].strip())
                
                optimal_solutions[instance] = {
                    'optimal_cost': objective,
                    'vehicles_used': vehicles,
                    'success': True
                }
            else:
                optimal_solutions[instance] = {
                    'optimal_cost': None,
                    'vehicles_used': None,
                    'success': False
                }
        
        return optimal_solutions
        
    except FileNotFoundError:
        print("Error: optimal_solutions.json not found. Run the optimal solver first.")
        return {}

def run_rl_model(instance, seed, model):
    """Run RL model and capture metrics"""
    start_time = time.time()
    
    try:
        # Import your RL modules here
        from test import evaluate
        from data import load_data
        
        # Load instance data
        instance_path = f"../dataset/files/{instance}"
        coords, demands, capacity, vehicles = load_data(instance_path)
        
        # Set seeds
        np.random.seed(seed)
        
        # Run RL evaluation - adjust parameters as needed
        results = evaluate(
            model=model,
            max_customers=100,  # Adjust based on your model
            test_file=instance_path,
            num_episodes=1,  # Single episode per run
            vehicle_cost=50.0,
            use_multi_run=False,
            wrapper_params=None  # Add if needed
        )
        
        computation_time = time.time() - start_time
        
        if results and len(results) > 0:
            result = results[0]  # Take first result
            
            return {
                "algorithm": "RL",
                "instance": instance,
                "seed": seed,
                "solution_cost": result['objective_cost'],
                "computation_time": computation_time,
                "num_vehicles": result['vehicles'],
                "routes": result.get('routes', []),
                "feasible": result['complete'],
                "success": True,
                "error": None
            }
        else:
            return {
                "algorithm": "RL",
                "instance": instance,
                "seed": seed,
                "solution_cost": float('inf'),
                "computation_time": computation_time,
                "num_vehicles": 0,
                "routes": [],
                "feasible": False,
                "success": False,
                "error": "No results returned"
            }
            
    except Exception as e:
        return {
            "algorithm": "RL",
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

def evaluate_rl(optimal_solutions, model):
    """Evaluate RL across all instances"""
    print("Evaluating RL Algorithm")
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
            
            result = run_rl_model(instance, seed, model)
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
        
        # Get optimal cost
        optimal_cost = None
        if instance in optimal_solutions and optimal_solutions[instance]['success']:
            optimal_cost = optimal_solutions[instance]['optimal_cost']
        
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
            
            # Optimality metrics - compare total cost to total cost
            deviation_from_optimal = None
            within_95_percent = 0.0
            within_99_percent = 0.0
            
            if optimal_cost:
                deviation_from_optimal = ((best_cost - optimal_cost) / optimal_cost) * 100
                within_95_percent = sum(1 for c in costs if c <= optimal_cost * 1.05) / len(costs)
                within_99_percent = sum(1 for c in costs if c <= optimal_cost * 1.01) / len(costs)
            
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
            deviation_from_optimal = float('inf') if optimal_cost else None
            within_95_percent = 0.0
            within_99_percent = 0.0
            robustness_cost = 0.0
            robustness_time = 0.0
        
        metrics.append({
            "algorithm": "RL",
            "instance": instance,
            "instance_type": instance[0].upper(),
            "instance_size": int(instance.split('_')[1].split('.')[0]),
            
            # Solution Quality
            "optimal_cost": optimal_cost,
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
    
    all_results = []
    best_routes = {}
    
    # Generate seeds
    np.random.seed(42)
    seeds = np.random.randint(1, 100000, NUM_RUNS).tolist()
    
    # Process each instance with its specific model
    for instance in optimal_solutions.keys():
        print(f"\nProcessing {instance}...")
        
        # Load appropriate model for this instance type
        model_path = get_model_path(instance)
        print(f"Loading model: {model_path}")
        
        try:
            model = SAC.load(model_path)
        except Exception as e:
            print(f"Error loading model for {instance}: {e}")
            continue
        
        # Run evaluation for this instance
        instance_results = []
        best_cost = float('inf')
        best_route = None
        
        for run_idx, seed in enumerate(seeds):
            print(f"  Run {run_idx + 1}/{NUM_RUNS}", end=" ")
            
            result = run_rl_model(instance, seed, model)
            instance_results.append(result)
            
            # Track best route
            if result['success'] and result['solution_cost'] < best_cost:
                best_cost = result['solution_cost']
                best_route = result.get('routes', [])
            
            if result['success']:
                print(f"✓ Cost: {result['solution_cost']:.2f}")
            else:
                print(f"✗ Failed")
        
        # Store best route
        if best_route:
            best_routes[instance] = {
                "cost": best_cost,
                "routes": best_route
            }
        
        all_results.extend(instance_results)
    
    # Calculate metrics and save results
    metrics_df = calculate_metrics(all_results, optimal_solutions)
    
    pd.DataFrame(all_results).to_csv("RL_detailed.csv", index=False)
    metrics_df.to_csv("RL_metrics.csv", index=False)
    
    with open("RL_best.json", "w") as f:
        json.dump(best_routes, f, separators=(',', ':'))
    
    print(f"\nResults saved: RL_detailed.csv, RL_metrics.csv, RL_best.json")
    return all_results, metrics_df, best_routes

if __name__ == "__main__":
    results, metrics, best_routes = main()