import os
import sys
import time
import numpy as np
import pandas as pd
import numpy.random as rnd
import json
from io import StringIO
from contextlib import redirect_stdout

from ALNS import main as alns_main

RUNS = 20

def load_optimal():
    try:
        with open("../optimal_solutions.json", "r") as f:
            data = json.load(f)
        
        optimal = {}
        for instance, result_data in data.items():
            if result_data['success']:
                result_text = result_data['result']
                
                objective = None
                vehicles = None
                for line in result_text.split('\n'):
                    if 'objective:' in line:
                        objective = float(line.split(':')[1].strip())
                    elif '# routes:' in line:
                        vehicles = int(line.split(':')[1].strip())
                
                optimal[instance] = {
                    'cost': objective,
                    'vehicles': vehicles,
                    'success': True
                }
            else:
                optimal[instance] = {
                    'cost': None,
                    'vehicles': None,
                    'success': False
                }
        
        return optimal
        
    except FileNotFoundError:
        return {}

def run_alns(instance, seed):
    start_time = time.time()
    
    np.random.seed(seed)
    rnd.seed(seed)
    
    original_input = input
    sys.modules['builtins'].input = lambda _: instance
    
    try:
        f = StringIO()
        with redirect_stdout(f):
            alns_main()
        
        computation_time = time.time() - start_time
        output = f.getvalue()
        
        cost = None
        vehicles = None
        routes = []
        feasible = True
        
        lines = output.split('\n')
        for line in lines:
            if "Final cost:" in line:
                cost = float(line.split(':')[1].strip())
            elif "Vehicles used:" in line:
                vehicles = int(line.split(':')[1].strip())
            elif "  Vehicle " in line and ": [" in line:
                try:
                    route_part = line.split(': [')[1].rstrip(']')
                    if route_part.strip():
                        route = [int(x.strip()) for x in route_part.split(',') if x.strip()]
                        if route:
                            routes.append([0] + route + [0]) 
                except:
                    pass
            elif "Unassigned:" in line:
                unassigned_part = line.split(':')[1].strip()
                if unassigned_part != "[]" and unassigned_part:
                    feasible = False
        
        return {
            "algorithm": "ALNS",
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
            "algorithm": "ALNS",
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
        sys.modules['builtins'].input = original_input

def evaluate(optimal):
    instances = list(optimal.keys())
    
    all_results = []
    best_routes = {}
    
    np.random.seed(42)
    seeds = np.random.randint(1, 100000, RUNS).tolist()
    
    for instance in instances:
        instance_results = []
        best_cost = float('inf')
        best_route = None
        
        for run_idx, seed in enumerate(seeds):
            result = run_alns(instance, seed)
            instance_results.append(result)

            if result['success'] and result['solution_cost'] < best_cost:
                best_cost = result['solution_cost']
                best_route = result.get('routes', [])
        
        if best_route:
            best_routes[instance] = {
                "cost": best_cost,
                "routes": best_route
            }
        
        all_results.extend(instance_results)
    
    return all_results, best_routes

def calc_metrics(results, optimal):
    df = pd.DataFrame(results)
    metrics = []
    
    for instance in df['instance'].unique():
        instance_data = df[df['instance'] == instance]
        successful_runs = instance_data[instance_data['success'] == True]
        
        opt_cost = None
        if instance in optimal and optimal[instance]['success']:
            opt_cost = optimal[instance]['cost']
        
        if len(successful_runs) > 0:
            costs = successful_runs['solution_cost']
            times = successful_runs['computation_time']
            vehicles = successful_runs['num_vehicles']
            feasible_runs = successful_runs[successful_runs['feasible'] == True]
            
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
            
            deviation = None
            within_95 = 0.0
            within_99 = 0.0
            
            if opt_cost:
                deviation = ((best_cost - opt_cost) / opt_cost) * 100
                within_95 = sum(1 for c in costs if c <= opt_cost * 1.05) / len(costs)
                within_99 = sum(1 for c in costs if c <= opt_cost * 1.01) / len(costs)
            
            robustness_cost = cost_std / avg_cost if avg_cost > 0 else 0
            robustness_time = time_std / avg_time if avg_time > 0 else 0
            
        else:
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
        
        metrics.append({
            "algorithm": "ALNS",
            "instance": instance,
            "instance_type": instance[0].upper(),
            "instance_size": int(instance.split('_')[1].split('.')[0]),
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
    optimal = load_optimal()
    if not optimal:
        return None, None, None
    
    results, best_routes = evaluate(optimal)
    
    metrics_df = calc_metrics(results, optimal)
    
    pd.DataFrame(results).to_csv("ALNS_detailed.csv", index=False)
    
    metrics_df.to_csv("ALNS_metrics.csv", index=False)
    
    with open("ALNS_best.json", "w") as f:
        json.dump(best_routes, f, separators=(',', ':'))
    
    return results, metrics_df, best_routes

if __name__ == "__main__":
    results, metrics, best_routes = main()