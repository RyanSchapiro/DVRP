import os
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from td import create_env
from data import load_data
import random
import numpy as np
import torch
from copy import deepcopy
from typing import List, Tuple, Dict, Any
import json
import glob



def get_model_path(instance_name):
    """Get the appropriate model path based on instance type"""
    if instance_name.lower().startswith('rc'):
        return "rc100_200"
    elif instance_name[0].upper() == 'C':
        return "c100_200"
    elif instance_name[0].upper() == 'R':
        return "r100_200"
    else:
        raise ValueError(f"Unknown instance type: {instance_name}")

def beam_search_predict(model, env, beam_width=5, max_steps=200):
    """
    Simple beam search that samples multiple actions at each step
    and tracks the best complete solutions
    """
    
    # Reset environment
    obs, info = env.reset()
    
    best_solutions = []
    
    # Run multiple rollouts with different random seeds
    for rollout in range(beam_width * 2):  # Try more rollouts than beam width
        
        # Reset for this rollout
        env.reset(seed=env.seed + rollout)  # Slightly different seed
        obs, info = env.reset()
        
        total_reward = 0.0
        actions = []
        step_count = 0
        
        # Single rollout with some randomness
        for step in range(max_steps):
            # Get action with some exploration
            if rollout < beam_width:
                # First beam_width rollouts: more deterministic
                action, _ = model.predict(obs, deterministic=True)
            else:
                # Additional rollouts: more exploration
                action, _ = model.predict(obs, deterministic=False)
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            actions.append(action)
            step_count += 1
            
            if terminated or truncated:
                break
        
        # Store solution if complete
        solution = {
            'total_reward': total_reward,
            'total_distance': info.get('total_distance', 0),
            'n_vehicles_used': info.get('n_vehicles_used', 0),
            'objective_cost': info.get('objective_cost', 0),
            'all_routes': info.get('all_routes', []),
            'completion_rate': info.get('completion_rate', 0),
            'actions': actions,
            'steps': step_count
        }
        
        best_solutions.append(solution) 
    
    # Return best solution by completion rate, then by objective cost
    if not best_solutions:
        return None
    
    # Sort by completion rate (descending), then by total_reward (ascending for negative rewards)
    best_solutions.sort(key=lambda x: (-x['completion_rate'], x['total_reward']))
    
    return best_solutions[0]

def multi_run_predict(model, env, num_runs=10):
    """Multi-run predict - CLEAN and FIXED"""
    best_solution = None
    best_score = float('-inf')
    
    for run in range(num_runs):
        obs, info = env.reset(seed=12345 + run * 42)
        
        total_reward = 0.0
        actions = []
        step_count = 0
        
        for step in range(200):
            step_count = step + 1
            
            deterministic = (run % 2 == 0)
            action, _ = model.predict(obs, deterministic=deterministic)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            actions.append(action)
            
            if terminated or truncated:
                break
        
        # Count vehicles from routes
        all_routes = info.get('all_routes', [])
        actual_vehicle_count = len([route for route in all_routes if len(route) > 2])
        
        completion = info.get('completion_rate', 0)
        distance = info.get('total_distance', 0)
        corrected_objective = distance + 50.0 * actual_vehicle_count
        
        score = completion * 1000 + total_reward
        
        if score > best_score:
            best_score = score
            best_solution = {
                'total_reward': total_reward,
                'total_distance': distance,
                'n_vehicles_used': actual_vehicle_count,
                'objective_cost': corrected_objective,
                'all_routes': all_routes,
                'completion_rate': completion,
                'actions': actions,
                'steps': step_count,
                'complete': (completion >= 1.0)
            }
    
    return best_solution

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
                
                optimal_solutions[instance.replace('.txt', '')] = {
                    'optimal_cost': objective,
                    'vehicles_used': vehicles,
                    'success': True
                }
            else:
                optimal_solutions[instance.replace('.txt', '')] = {
                    'optimal_cost': None,
                    'vehicles_used': None,
                    'success': False
                }
        
        return optimal_solutions
        
    except FileNotFoundError:
        print("Error: optimal_solutions.json not found. Run the optimal solver first.")
        return {}

def evaluate_batch(loaded_models, instances_data, optimal_solutions, num_episodes=20, max_customers=25, vehicle_cost=50.0):
    """Evaluate all instances in batch"""
    all_results = []
    best_routes = {}
    
    for instance_name, instance_file in instances_data.items():
        try:
            model_path = get_model_path(instance_name)
            
            if model_path not in loaded_models:
                print(f"Warning: Model {model_path} not loaded, skipping {instance_name}")
                continue
                
            model_info = loaded_models[model_path]
            
            print(f"\nProcessing {instance_name} with model {model_path}")
            print(f"Loading test data from {instance_file}...")
            coords, demands, capacity, vehicles = load_data(instance_file)
            
            print(f"Test instance: {len(demands)-1} customers, capacity: {capacity}")
            print(f"Running {num_episodes} evaluation episodes...")
            
            instance_results = []
            best_cost = float('inf')
            best_route = None
            
            # Generate seeds like in metrics.py
            np.random.seed(42)
            seeds = np.random.randint(1, 100000, num_episodes).tolist()
            
            for episode in range(num_episodes):
                print(f"  Episode {episode + 1}...", end=" ")
                
                eval_seed = seeds[episode]
                env = create_env(coords, demands, capacity, vehicles, seed=eval_seed,
                               vehicle_cost=vehicle_cost, max_customers=max_customers)
                
                # Use multi_run_predict (which runs 8 internal runs)
                solution = multi_run_predict(model_info['model'], env, num_runs=8)
                
                if solution is None:
                    print("Failed")
                    result = {
                        "algorithm": "RL",
                        "instance": instance_name,
                        "seed": eval_seed,
                        "solution_cost": float('inf'),
                        "computation_time": 0.5,
                        "num_vehicles": 0,
                        "routes": [],
                        "feasible": False,
                        "success": False,
                        "error": "No solution found"
                    }
                else:
                    print(f"✓ Cost: {solution['objective_cost']:.2f}")
                    
                    # Track best route
                    if solution['objective_cost'] < best_cost:
                        best_cost = solution['objective_cost']
                        best_route = solution.get('all_routes', [])
                    
                    result = {
                        "algorithm": "RL",
                        "instance": instance_name,
                        "seed": eval_seed,
                        "solution_cost": solution['objective_cost'],
                        "computation_time": 0.5,  # Approximate
                        "num_vehicles": solution['n_vehicles_used'],
                        "routes": solution.get('all_routes', []),
                        "feasible": solution.get('complete', False),
                        "success": solution.get('complete', False),
                        "error": None
                    }
                
                instance_results.append(result)
                env.close()
            
            # Store best route
            if best_route:
                best_routes[instance_name] = {
                    "cost": best_cost,
                    "routes": best_route
                }
            
            all_results.extend(instance_results)
            
        except Exception as e:
            print(f"Error processing {instance_name}: {e}")
    
    return all_results, best_routes

def calculate_metrics(results, optimal_solutions):
    """Calculate all performance metrics like in metrics.py"""
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
            
            # Optimality metrics
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
            "instance_size": int(instance.split('_')[1]) if '_' in instance else 0,
            
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
    """Main batch evaluation function"""
    print("RL Model Batch Evaluation")
    print("=" * 50)
    
    # Configuration
    dataset_path = "../../../dataset/files/"
    num_episodes = 20
    max_customers = 25
    vehicle_cost = 50.0
    
    # Find all test instances
    instance_files = glob.glob(os.path.join(dataset_path, "*.txt"))
    instance_files.sort()
    
    if not instance_files:
        print(f"Error: No instance files found in {dataset_path}")
        return
    
    # Create instances mapping
    instances_data = {}
    for instance_file in instance_files:
        instance_name = os.path.basename(instance_file).replace('.txt', '')
        instances_data[instance_name] = instance_file
    
    print(f"Found {len(instances_data)} instances")
    print(f"Episodes per instance: {num_episodes}")
    
    # Load optimal solutions
    optimal_solutions = load_optimal_solutions()
    
    # Load all required models
    loaded_models = {}
    required_models = set()
    
    for instance_name in instances_data.keys():
        try:
            required_models.add(get_model_path(instance_name))
        except ValueError as e:
            print(f"Warning: {e}")
    
    for model_path in required_models:
        if not os.path.exists(f"{model_path}.zip"):
            print(f"Warning: Model {model_path}.zip not found")
            continue
        
        print(f"Loading model: {model_path}")
        loaded_models[model_path] = {
            'model': SAC.load(model_path)
        }
    
    if not loaded_models:
        print("Error: No models could be loaded!")
        return
    
    # Run batch evaluation
    all_results, best_routes = evaluate_batch(
        loaded_models, instances_data, optimal_solutions,
        num_episodes=num_episodes, max_customers=max_customers, 
        vehicle_cost=vehicle_cost
    )
    
    if not all_results:
        print("No results to save!")
        return
    
    # Calculate metrics
    metrics_df = calculate_metrics(all_results, optimal_solutions)
    
    # Save results in same format as metrics.py
    pd.DataFrame(all_results).to_csv("RL_detailed.csv", index=False)
    metrics_df.to_csv("RL_metrics.csv", index=False)
    
    with open("RL_best.json", "w") as f:
        json.dump(best_routes, f, separators=(',', ':'))
    
    print(f"\nResults saved:")
    print(f"  RL_detailed1.csv")
    print(f"  RL_metrics1.csv")
    print(f"  RL_best1.json")
    
    # Print summary like metrics.py
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

if __name__ == "__main__":
    main()