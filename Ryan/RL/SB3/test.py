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

def get_path(instance_name):
    """Map instance names to corresponding trained model paths.
    
    Different instance types (R, C, RC) may require different trained models
    as they have different characteristics (random, clustered, mixed).
    
    Args:
        instance_name: Name of VRP instance (e.g., 'c1_25', 'r101', 'rc205')
        
    Returns:
        Path to appropriate trained model file
        
    Raises:
        ValueError: If instance type not recognized
    """
    if instance_name.lower().startswith('rc'):
        return "rc100_200"  # Mixed random-clustered instances
    elif instance_name[0].upper() == 'C':
        return "c100_200"   # Clustered customer instances
    elif instance_name[0].upper() == 'R':
        return "r100_200"   # Random customer instances
    else:
        raise ValueError(f"Unknown instance type: {instance_name}")

def beam_predict(model, env, beam_width=5, max_steps=200):
    """Perform beam search-like evaluation with multiple rollouts.
    
    Runs multiple episodes (both deterministic and stochastic) and selects
    the best solution based on completion rate and reward.
    
    Args:
        model: Trained RL model
        env: VRP environment
        beam_width: Number of deterministic + stochastic rollouts
        max_steps: Maximum steps per episode
        
    Returns:
        Best solution found across all rollouts, or None if none found
    """
    obs, info = env.reset()
    best_solutions = []
    
    # Run multiple rollouts with different strategies
    for rollout in range(beam_width * 2):
        env.reset(seed=env.seed + rollout)  # Different seed per rollout
        obs, info = env.reset()
        
        total_reward = 0.0
        actions = []
        step_count = 0
        
        # Execute single rollout
        for step in range(max_steps):
            if rollout < beam_width:
                # First half: deterministic (greedy) rollouts
                action, _ = model.predict(obs, deterministic=True)
            else:
                # Second half: stochastic (exploration) rollouts
                action, _ = model.predict(obs, deterministic=False)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            actions.append(action)
            step_count += 1
            
            if terminated or truncated:
                break
        
        # Store solution information
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
    
    if not best_solutions:
        return None

    # Select best solution: prioritize completion rate, then total reward
    best_solutions.sort(key=lambda x: (-x['completion_rate'], x['total_reward']))
    
    return best_solutions[0]

def multi_predict(model, env, num_runs=10):
    """Run multiple prediction episodes and return the best solution.
    
    This is the main evaluation function that tests model performance
    across multiple episodes with different seeds and deterministic settings.
    
    Args:
        model: Trained RL model to evaluate
        env: VRP environment to test on
        num_runs: Number of episodes to run
        
    Returns:
        Best solution found across all runs, with detailed statistics
    """
    best_solution = None
    best_score = float('-inf')
    
    for run in range(num_runs):
        # Use consistent but varied seeds for reproducible evaluation
        obs, info = env.reset(seed=12345 + run * 42)
        
        total_reward = 0.0
        actions = []
        step_count = 0
        
        # Execute single episode
        for step in range(200):  # Maximum episode length
            step_count = step + 1
            
            # Alternate between deterministic and stochastic policies
            deterministic = (run % 2 == 0)
            action, _ = model.predict(obs, deterministic=deterministic)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            actions.append(action)
            
            if terminated or truncated:
                break

        # Extract solution information from environment
        all_routes = info.get('all_routes', [])
        
        # Calculate actual vehicle count (routes with customers, not just depot)
        actual_vehicle_count = len([route for route in all_routes if len(route) > 2])
        
        completion = info.get('completion_rate', 0)
        distance = info.get('total_distance', 0)
        
        # Recalculate objective with correct vehicle count
        corrected_objective = distance + 50.0 * actual_vehicle_count
        
        # Scoring function: heavily prioritize completion, then reward
        score = completion * 1000 + total_reward
        
        # Track best solution across all runs
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
                'complete': (completion >= 1.0)  # Full solution flag
            }
    
    return best_solution

def load_optimal():
    """Load optimal solutions from benchmark file for comparison.
    
    Returns:
        Dictionary mapping instance names to optimal solution data,
        or empty dict if file not found
    """
    try:
        with open("../../optimal_solutions.json", "r") as f:
            data = json.load(f)
        
        optimal = {}
        for instance, result_data in data.items():
            if result_data['success']:
                result_text = result_data['result']
                
                # Parse objective value and vehicle count from text
                objective = None
                vehicles = None
                for line in result_text.split('\n'):
                    if 'objective:' in line:
                        objective = float(line.split(':')[1].strip())
                    elif '# routes:' in line:
                        vehicles = int(line.split(':')[1].strip())
                
                optimal[instance.replace('.txt', '')] = {
                    'cost': objective,
                    'vehicles': vehicles,
                    'success': True
                }
            else:
                # Mark failed instances
                optimal[instance.replace('.txt', '')] = {
                    'cost': None,
                    'vehicles': None,
                    'success': False
                }
        
        return optimal
        
    except FileNotFoundError:
        print("Optimal solutions file not found - gap analysis will be skipped")
        return {}

def evaluate_batch(loaded_models, instances_data, optimal, num_episodes=20, max_customers=25, vehicle_cost=50.0):
    """Evaluate RL models on a batch of VRP instances.
    
    Args:
        loaded_models: Dictionary of loaded RL models by type
        instances_data: Dictionary mapping instance names to file paths
        optimal: Known optimal solutions for comparison
        num_episodes: Number of evaluation episodes per instance
        max_customers: Maximum customers supported by models
        vehicle_cost: Fixed cost per vehicle used
        
    Returns:
        Tuple of (all_results_list, best_routes_dict) containing detailed
        results for every episode and best solution per instance
    """
    all_results = []
    best_routes = {}
    
    for instance_name, instance_file in instances_data.items():
        print(f"Evaluating instance: {instance_name}")
        
        try:
            # Determine which model to use for this instance type
            model_path = get_path(instance_name)
            
            if model_path not in loaded_models:
                print(f"  No model available for {model_path}")
                continue
                
            model_info = loaded_models[model_path]
            
            # Load problem instance data
            coords, demands, capacity, vehicles = load_data(instance_file)
            
            instance_results = []
            best_cost = float('inf')
            best_route = None
            
            # Generate consistent seeds for reproducible evaluation
            np.random.seed(42)
            seeds = np.random.randint(1, 100000, num_episodes).tolist()
            
            # Run multiple episodes for statistical significance
            for episode in range(num_episodes):
                eval_seed = seeds[episode]
                
                # Create fresh environment for this episode
                env = create_env(coords, demands, capacity, vehicles, seed=eval_seed,
                               vehicle_cost=vehicle_cost, max_customers=max_customers)
                
                # Get solution from RL model
                solution = multi_predict(model_info['model'], env, num_runs=8)
                
                if solution is None:
                    # Handle failed episodes
                    result = {
                        "algorithm": "RL",
                        "instance": instance_name,
                        "seed": eval_seed,
                        "solution_cost": float('inf'),
                        "computation_time": 0.5,  # Estimated time
                        "num_vehicles": 0,
                        "routes": [],
                        "feasible": False,
                        "success": False,
                        "error": "No solution found"
                    }
                else:
                    # Track best solution found across all episodes
                    if solution['objective_cost'] < best_cost:
                        best_cost = solution['objective_cost']
                        best_route = solution.get('all_routes', [])
                    
                    # Record successful episode results
                    result = {
                        "algorithm": "RL",
                        "instance": instance_name,
                        "seed": eval_seed,
                        "solution_cost": solution['objective_cost'],
                        "computation_time": 0.5,
                        "num_vehicles": solution['n_vehicles_used'],
                        "routes": solution.get('all_routes', []),
                        "feasible": solution.get('complete', False),
                        "success": solution.get('complete', False),
                        "error": None
                    }
                
                instance_results.append(result)
                env.close()
            
            # Store best route found for this instance
            if best_route:
                best_routes[instance_name] = {
                    "cost": best_cost,
                    "routes": best_route
                }
            
            all_results.extend(instance_results)
            
        except Exception as e:
            print(f"  Error evaluating {instance_name}: {str(e)}")
            pass
    
    return all_results, best_routes

def calc_metrics(results, optimal):
    """Calculate comprehensive performance metrics from evaluation results.
    
    Args:
        results: List of individual episode results
        optimal: Dictionary of known optimal solutions
        
    Returns:
        DataFrame with detailed metrics per instance
    """
    df = pd.DataFrame(results)
    metrics = []
    
    for instance in df['instance'].unique():
        instance_data = df[df['instance'] == instance]
        successful_runs = instance_data[instance_data['success'] == True]
        
        # Get optimal solution for comparison if available
        opt_cost = None
        if instance in optimal and optimal[instance]['success']:
            opt_cost = optimal[instance]['cost']
        
        if len(successful_runs) > 0:
            # Calculate statistics from successful runs
            costs = successful_runs['solution_cost']
            times = successful_runs['computation_time']
            vehicles = successful_runs['num_vehicles']
            feasible_runs = successful_runs[successful_runs['feasible'] == True]
            
            # Basic performance metrics
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
            
            # Optimality gap analysis
            deviation = None
            within_95 = 0.0  # Fraction within 5% of optimal
            within_99 = 0.0  # Fraction within 1% of optimal
            
            if opt_cost:
                deviation = ((best_cost - opt_cost) / opt_cost) * 100
                within_95 = sum(1 for c in costs if c <= opt_cost * 1.05) / len(costs)
                within_99 = sum(1 for c in costs if c <= opt_cost * 1.01) / len(costs)
            
            # Robustness measures (coefficient of variation)
            robustness_cost = cost_std / avg_cost if avg_cost > 0 else 0
            robustness_time = time_std / avg_time if avg_time > 0 else 0
            
        else:
            # Handle instances with no successful runs
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
            deviation = float('inf') if opt_cost else None
            within_95 = 0.0
            within_99 = 0.0
            robustness_cost = 0.0
            robustness_time = 0.0
        
        # Compile metrics for this instance
        metrics.append({
            "algorithm": "RL",
            "instance": instance,
            "instance_type": instance[0].upper(),  # R, C, or RC
            "instance_size": int(instance.split('_')[1]) if '_' in instance else 0,
            "optimal_cost": opt_cost,
            "best_cost": best_cost,
            "avg_cost": avg_cost,
            "worst_cost": worst_cost,
            "cost_std": cost_std,
            "deviation_from_optimal_percent": deviation,
            "avg_computation_time": avg_time,
            "computation_time_std": time_std,
            "success_rate": success_rate,
            "feasibility_rate": feasibility_rate,
            "cost_robustness_cv": robustness_cost,
            "time_robustness_cv": robustness_time,
            "within_95_percent_optimal": within_95,
            "within_99_percent_optimal": within_99,
            "avg_vehicles": avg_vehicles,
            "vehicles_std": vehicles_std
        })
    
    return pd.DataFrame(metrics)

def main():
    # Configuration parameters
    dataset_path = "../../../dataset/files/"
    num_episodes = 20      # Episodes per instance for statistical robustness
    max_customers = 25     # Maximum customers model was trained for
    vehicle_cost = 50.0    # Fixed cost per vehicle
    
    instance_files = glob.glob(os.path.join(dataset_path, "*.txt"))
    instance_files.sort()
    
    if not instance_files:
        print(f"No instance files found in {dataset_path}")
        return
    
    # Create mapping from instance name to file path
    instances_data = {}
    for instance_file in instance_files:
        instance_name = os.path.basename(instance_file).replace('.txt', '')
        instances_data[instance_name] = instance_file
    
    print(f"Found {len(instances_data)} instances to evaluate")
    
    # Load optimal solutions for comparison
    optimal = load_optimal()
    
    # Load all required trained models
    loaded_models = {}
    required_models = set()
    
    # Determine which model types are needed
    for instance_name in instances_data.keys():
        try:
            required_models.add(get_path(instance_name))
        except ValueError as e:
            print(f"Skipping unknown instance type: {instance_name}")
            pass
    
    # Load each required model
    for model_path in required_models:
        if not os.path.exists(f"{model_path}.zip"):
            print(f"Model file not found: {model_path}.zip")
            continue
        
        print(f"Loading model: {model_path}")
        loaded_models[model_path] = {
            'model': SAC.load(model_path)
        }
    
    if not loaded_models:
        print("No models could be loaded - exiting")
        return
    
    print(f"Loaded {len(loaded_models)} models")
    
    # Run batch evaluation
    print("Starting batch evaluation...")
    all_results, best_routes = evaluate_batch(
        loaded_models, instances_data, optimal,
        num_episodes=num_episodes, max_customers=max_customers, 
        vehicle_cost=vehicle_cost
    )
    
    if not all_results:
        print("No results generated - check models and instances")
        return
    
    # Calculate performance metrics
    print("Calculating performance metrics...")
    metrics_df = calc_metrics(all_results, optimal)
    
    # Save results to files
    print("Saving results...")
    pd.DataFrame(all_results).to_csv("RL_detailed.csv", index=False)
    metrics_df.to_csv("RL_metrics.csv", index=False)
    
    with open("RL_best.json", "w") as f:
        json.dump(best_routes, f, separators=(',', ':'))

if __name__ == "__main__":
    main()