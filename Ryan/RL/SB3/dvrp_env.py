import numpy as np
import gymnasium as gym
from gymnasium import spaces


MAX_CUSTOMERS = 100  # Universal constant
class VRP(gym.Env):

    def __init__(self, coords, demands, capacity, max_vehicles=None, seed=42, 
                 static_ratio=0.7, arrival_rate=0.4):
        super().__init__()
        
        # Problem data
        self.coords = coords.copy() 
        self.demands = demands.copy()
        self.capacity = float(capacity)
        self.n_customers = len(demands) - 1  # Exclude depot
        self.seed = seed
        
        self.static_ratio = static_ratio 
        self.arrival_rate = arrival_rate 
        
        # Max vehicles (reasonable default)
        if max_vehicles is None:
            total_demand = np.sum(demands[1:])
            self.max_vehicles = max(1, int(np.ceil(total_demand / capacity)) + 2)
        else:
            self.max_vehicles = max_vehicles
        

        self.dist_matrix = self._compute_distances()
        
        self.action_space = spaces.Discrete(self.n_customers + 1)
        

        self.max_customers_universal = MAX_CUSTOMERS
        obs_dim = 4 + 5 * MAX_CUSTOMERS  # Always 504 elements
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        # Normalization constants
        self.max_coord = np.max(np.abs(coords))
        self.max_demand = np.max(demands) if len(demands) > 1 else 1
        
        self.reset()
    
    def _compute_distances(self):
        distances = np.linalg.norm(self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :], axis=-1)
        return distances.astype(int)
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed = seed
        np.random.seed(self.seed)
        
        # Generate stochastic requests
        self._generate_stochastic_requests()
        

        self.current_pos = 0  # Start at depot
        self.remaining_capacity = self.capacity
        self.visited = np.zeros(len(self.demands), dtype=bool)
        self.visited[0] = True
        self.time = 0
        
        # Solution tracking
        self.total_distance = 0.0
        self.current_route = [0]
        self.all_routes = []
        self.n_vehicles_used = 0
        
        return self._get_obs(), self._get_info()
    
    def _generate_stochastic_requests(self):
        from data import Poisson
        
        static_customers, dynamic_arrivals = Poisson(
            dim=len(self.demands),
            stat=self.static_ratio,
            lam=self.arrival_rate,
            seed=self.seed
        )
        

        self.revealed = np.zeros(len(self.demands), dtype=bool)
        self.revealed[0] = True  # Depot always revealed
        
        for customer in static_customers:
            if customer < len(self.demands):
                self.revealed[customer] = True
        
        self.arrival_times = dynamic_arrivals
    
    def step(self, action):
        self.time += 1
        self._reveal_dynamic_customers()
        
        if action >= len(self.demands):
            action = 0
        
        if not self._is_valid_action(action):
            if self._is_valid_action(0):
                action = 0
            else:
                return self._get_obs(), -1000, True, False, self._get_info()
        
        step_distance = self.dist_matrix[self.current_pos, action]
        self.total_distance += step_distance
        
        self.current_pos = action
        self.current_route.append(action)
        
        reward = -step_distance
        
        if action == 0:
            route_has_customers = any(node != 0 for node in self.current_route[1:-1])
            
            if route_has_customers:
                self.all_routes.append(self.current_route.copy())
            
            self.current_route = [0]
            self.remaining_capacity = self.capacity
            
        else:
            self.visited[action] = True
            self.remaining_capacity -= self.demands[action]
        
        unvisited_revealed = self._get_unvisited_revealed()
        
        if not np.any(unvisited_revealed):
            future_arrivals = any(
                arrival_time > self.time 
                for arrival_time in self.arrival_times.values()
            )
            
            if not future_arrivals:
                if self.current_pos != 0:
                    final_distance = self.dist_matrix[self.current_pos, 0]
                    self.total_distance += final_distance
                    reward -= final_distance
                    self.current_route.append(0)
                
                final_route_has_customers = any(node != 0 for node in self.current_route[1:-1])
                if final_route_has_customers:
                    self.all_routes.append(self.current_route.copy())

                terminated = True
            else:
                terminated = False
        else:
            terminated = False
        
        valid_actions = self._get_action_mask()
        if not np.any(valid_actions) and not terminated:
            reward -= 100
            terminated = True
        
        return self._get_obs(), reward, terminated, False, self._get_info()
    
    def _reveal_dynamic_customers(self):
        for customer, arrival_time in self.arrival_times.items():
            if self.time >= arrival_time and not self.revealed[customer]:
                self.revealed[customer] = True
    
    def _get_unvisited_revealed(self):
        unvisited_revealed = self.revealed & ~self.visited
        unvisited_revealed[0] = False  # Exclude depot
        return unvisited_revealed
    
    def _get_obs(self):
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # State features (first 4 elements)
        obs[0] = self.coords[self.current_pos, 0] / self.max_coord
        obs[1] = self.coords[self.current_pos, 1] / self.max_coord  
        obs[2] = self.remaining_capacity / self.capacity
        obs[3] = min(self.time / 100.0, 1.0)
        
        # Customer features (pad with zeros for unused slots)
        for i in range(MAX_CUSTOMERS):
            base_idx = 4 + i * 5
            
            if i < self.n_customers:  # Real customer
                customer_idx = i + 1
                obs[base_idx] = self.coords[customer_idx, 0] / self.max_coord
                obs[base_idx + 1] = self.coords[customer_idx, 1] / self.max_coord
                obs[base_idx + 2] = self.demands[customer_idx] / self.max_demand
                obs[base_idx + 3] = 1.0 if self.revealed[customer_idx] else 0.0
                obs[base_idx + 4] = 1.0 if self.visited[customer_idx] else 0.0
            else:  # Padding (already zeros)
                pass  # obs[base_idx:base_idx+5] remain 0
        
        return obs
    
    def _is_valid_action(self, action):
        if action == 0:  # Depot
            return True 
        
        if action >= len(self.demands):
            return False
        
        return (self.revealed[action] and 
                not self.visited[action] and 
                self.demands[action] <= self.remaining_capacity)
    
    def _get_action_mask(self):
        mask = np.zeros(MAX_CUSTOMERS + 1, dtype=bool)  # +1 for depot
        
        # Depot always valid
        mask[0] = True
        
        # Only real customers can be valid
        for i in range(min(self.n_customers, MAX_CUSTOMERS)):
            customer_idx = i + 1
            if customer_idx < len(self.demands):
                mask[i + 1] = self._is_valid_action(customer_idx)
        
        return mask
    

    def _get_info(self):
        unvisited_revealed = self._get_unvisited_revealed()
        total_unvisited = np.sum(~self.visited[1:])
        revealed_unvisited = np.sum(unvisited_revealed)
        

        completed_vehicles = len(self.all_routes)
        current_has_customers = any(node != 0 for node in self.current_route[1:])
        total_vehicles = completed_vehicles + (1 if current_has_customers else 0)
        
        vehicle_cost = 50.0
        full_objective = self.total_distance + vehicle_cost * total_vehicles
        
        return {
            'total_distance': self.total_distance,
            'n_vehicles_used': total_vehicles,
            'customers_served': self.n_customers - total_unvisited,
            'customers_remaining': total_unvisited,
            'revealed_remaining': revealed_unvisited,
            'remaining_capacity': self.remaining_capacity,
            'current_route': self.current_route.copy(),
            'all_routes': [route.copy() for route in self.all_routes],
            'action_mask': self._get_action_mask(),
            'completion_rate': (self.n_customers - total_unvisited) / self.n_customers,
            'time': self.time,
            'revealed': self.revealed.copy(),
            'arrival_times': self.arrival_times.copy(),
            'objective_cost': full_objective,
            'complete': total_unvisited == 0
        }