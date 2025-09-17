import os
import sys
import time
import numpy as np
import pandas as pd
import numpy.random as rnd
import json
from io import StringIO
from contextlib import redirect_stdout
from CW import main as cw_main

RUNS = 20

def load_optimal():
    with open("optimal_solutions.json", "r") as f:
        data = json.load(f)
    
    optimal = {}
    for instance, result_data in data.items():
        if result_data['success']:
            text = result_data['result']
            cost = distance = vehicles = None
            
            for line in text.split('\n'):
                if 'objective:' in line:
                    cost = float(line.split(':')[1].strip())
                elif 'distance:' in line:
                    distance = float(line.split(':')[1].strip())
                elif '# routes:' in line:
                    vehicles = int(line.split(':')[1].strip())
            
            optimal[instance] = {
                'cost': cost,
                'distance': distance,
                'vehicles': vehicles,
                'success': True
            }
        else:
            optimal[instance] = {
                'cost': None,
                'distance': None,
                'vehicles': None,
                'success': False
            }
    
    return optimal

def run_cw(instance, seed):
    start = time.time()
    
    np.random.seed(seed)
    rnd.seed(seed)
    
    original_input = input
    sys.modules['builtins'].input = lambda _: instance
    
    try:
        f = StringIO()
        with redirect_stdout(f):
            cw_main()
        
        time_taken = time.time() - start
        output = f.getvalue()
        
        cost = vehicles = None
        routes = []
        feasible = True
        
        for line in output.split('\n'):
            if "Total Cost:" in line:
                cost = float(line.split(':')[1].strip())
            elif "Number of vehicles used:" in line:
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
            elif "  Unassigned customers:" in line:
                unassigned = line.split(':')[1].strip()
                if unassigned != "[]" and unassigned:
                    feasible = False
        
        return {
            "instance": instance,
            "seed": seed,
            "cost": cost if cost else float('inf'),
            "time": time_taken,
            "vehicles": vehicles if vehicles else 0,
            "routes": routes,
            "feasible": feasible,
            "success": cost is not None
        }
        
    except Exception as e:
        return {
            "instance": instance,
            "seed": seed,
            "cost": float('inf'),
            "time": time.time() - start,
            "vehicles": 0,
            "routes": [],
            "feasible": False,
            "success": False
        }
    finally:
        sys.modules['builtins'].input = original_input

def evaluate(optimal):
    instances = list(optimal.keys())
    results = []
    best_routes = {}
    
    np.random.seed(42)
    seeds = np.random.randint(1, 100000, RUNS).tolist()
    
    for instance in instances:
        best_cost = float('inf')
        best_route = None
        
        for seed in seeds:
            result = run_cw(instance, seed)
            results.append(result)
            
            if result['success'] and result['cost'] < best_cost:
                best_cost = result['cost']
                best_route = result.get('routes', [])
        
        if best_route:
            best_routes[instance] = {
                "cost": best_cost,
                "routes": best_route
            }
    
    return results, best_routes

def calc_metrics(results, optimal):
    df = pd.DataFrame(results)
    metrics = []
    
    for instance in df['instance'].unique():
        data = df[df['instance'] == instance]
        success = data[data['success'] == True]
        
        opt_cost = opt_distance = None
        if instance in optimal and optimal[instance]['success']:
            opt_cost = optimal[instance]['cost']
            opt_distance = optimal[instance]['distance']
        
        if len(success) > 0:
            costs = success['cost']
            times = success['time']
            vehicles = success['vehicles']
            feasible = success[success['feasible'] == True]
            
            success_rate = len(success) / len(data)
            feasibility_rate = len(feasible) / len(data)
            best_cost = costs.min()
            avg_cost = costs.mean()
            worst_cost = costs.max()
            cost_std = costs.std()
            avg_time = times.mean()
            time_std = times.std()
            avg_vehicles = vehicles.mean()
            
            deviation = None
            within_95 = within_99 = 0.0
            
            distances = costs - (vehicles * 50)
            
            if opt_distance:
                best_distance = distances.min()
                deviation = ((best_distance - opt_distance) / opt_distance) * 100
                within_95 = sum(1 for c in distances if c <= opt_distance * 1.05) / len(distances)
                within_99 = sum(1 for c in distances if c <= opt_distance * 1.01) / len(distances)
            
            cost_cv = cost_std / avg_cost if avg_cost > 0 else 0
            time_cv = time_std / avg_time if avg_time > 0 else 0
            
        else:
            success_rate = feasibility_rate = 0.0
            best_cost = avg_cost = worst_cost = float('inf')
            cost_std = 0.0
            avg_time = data['time'].mean()
            time_std = data['time'].std()
            avg_vehicles = 0.0
            deviation = float('inf') if opt_distance else None
            within_95 = within_99 = 0.0
            cost_cv = time_cv = 0.0
        
        metrics.append({
            "instance": instance,
            "type": instance[0].upper(),
            "size": int(instance.split('_')[1].split('.')[0]),
            "opt_cost": opt_cost,
            "opt_distance": opt_distance,
            "best_cost": best_cost,
            "avg_cost": avg_cost,
            "worst_cost": worst_cost,
            "cost_std": cost_std,
            "deviation": deviation,
            "avg_time": avg_time,
            "time_std": time_std,
            "success_rate": success_rate,
            "feasibility_rate": feasibility_rate,
            "cost_cv": cost_cv,
            "time_cv": time_cv,
            "within_95": within_95,
            "within_99": within_99,
            "avg_vehicles": avg_vehicles
        })
    
    return pd.DataFrame(metrics)

def main():
    optimal = load_optimal()
    results, best_routes = evaluate(optimal)
    metrics = calc_metrics(results, optimal)
    
    pd.DataFrame(results).to_csv("cw_results.csv", index=False)
    metrics.to_csv("cw_metrics.csv", index=False)
    
    with open("cw_routes.json", "w") as f:
        json.dump(best_routes, f, separators=(',', ':'))
    
    return results, metrics, best_routes

if __name__ == "__main__":
    results, metrics, best_routes = main()