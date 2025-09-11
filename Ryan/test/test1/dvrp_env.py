import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any


class SCVRPEnv(gym.Env):
    """
    SCVRP Environment with padding and original coordinate distances
    """
    
    def __init__(
        self,
        coords: np.ndarray,
        demands: np.ndarray, 
        capacity: float,
        vehicle_cost: float = 50.0,
        static_ratio: float = 0.7,
        arrival_rate: float = 0.4,
        max_customers: int = 100,
        seed: int = None
    ):
        super().__init__()
        
        # Store original problem size
        self.original_coords = coords.copy().astype(np.float32)
        self.original_demands = demands.copy().astype(np.float32)
        self.original_n_nodes = len(coords)
        self.original_n_customers = self.original_n_nodes - 1
        
        # Pad to fixed maximum size
        self.max_customers = max_customers
        self.capacity = float(capacity)
        self.vehicle_cost = vehicle_cost
        self.seed_value = seed
        
        # Dynamic arrival parameters
        self.static_ratio = static_ratio
        self.arrival_rate = arrival_rate
        
        # Create padded arrays
        self.n_nodes = max_customers + 1  # Fixed size: depot + max customers
        self.n_customers = max_customers   # Fixed size
        
        # Pad coordinates - use depot coordinates for padding
        self.coords = np.zeros((self.n_nodes, 2), dtype=np.float32)
        self.coords[:self.original_n_nodes] = self.original_coords
        # Fill padding positions with depot coordinates (they'll be masked)
        for i in range(self.original_n_nodes, self.n_nodes):
            self.coords[i] = self.original_coords[0]
        
        # Pad demands - padding positions get zero demand
        self.demands = np.zeros(self.n_nodes, dtype=np.float32)
        self.demands[:self.original_n_nodes] = self.original_demands
        
        # Precompute distance matrix (using original coordinates, no normalization)
        self.distance_matrix = self._compute_distance_matrix()
        
        # Fixed observation and action spaces
        self.action_space = spaces.Discrete(self.n_nodes)
        
        obs_size = (
            2 +  # depot coordinates
            self.n_customers * 4 +  # customer features (x, y, demand, revealed)
            3    # vehicle state (current_node, capacity, time)
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        print(f"Created environment: {self.original_n_customers} real customers, "
              f"padded to {self.n_customers}, obs shape: {obs_size}")
        
        self.reset()
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """Compute Euclidean distance matrix using original coordinates"""
        distances = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
        
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    distances[i, j] = np.linalg.norm(self.coords[i] - self.coords[j])
        
        return distances
    
    def _generate_arrivals(self):
        """Generate dynamic arrivals for real customers only"""
        from data import Poisson
        
        # Only generate arrivals for real customers
        static_customers, dynamic_arrivals = Poisson(
            dim=self.original_n_nodes,  # Only real nodes
            stat=self.static_ratio,
            lam=self.arrival_rate,
            seed=self.seed_value
        )
        
        # Initialize revealed mask
        self.revealed = np.zeros(self.n_nodes, dtype=bool)
        self.revealed[0] = True  # Depot always revealed
        
        # Mark static customers as revealed (only real customers)
        for customer_id in static_customers:
            if 1 <= customer_id < self.original_n_nodes:
                self.revealed[customer_id] = True
        
        # Store arrival times (only real customers)
        self.arrival_times = {}
        for customer_id, arrival_time in dynamic_arrivals.items():
            if 1 <= customer_id < self.original_n_nodes:
                self.arrival_times[customer_id] = max(1, int(arrival_time))
        
        # Padding positions are never revealed
        for i in range(self.original_n_nodes, self.n_nodes):
            self.revealed[i] = False
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        super().reset(seed=seed)
        
        if seed is not None:
            self.seed_value = seed
        
        # Generate arrivals
        self._generate_arrivals()
        
        # Reset state
        self.time_step = 0
        self.current_location = 0
        self.current_capacity = self.capacity
        self.visited = np.zeros(self.n_nodes, dtype=bool)
        self.visited[0] = True  # Depot is visited
        
        # Padding positions are always "visited" (masked out)
        for i in range(self.original_n_nodes, self.n_nodes):
            self.visited[i] = True
        
        # Solution tracking
        self.total_distance = 0.0
        self.current_route = [0]
        self.completed_routes = []
        self.n_vehicles_used = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute step"""
        
        # Validate action bounds
        if not (0 <= action < self.n_nodes):
            return self._get_observation(), -1000.0, True, False, self._get_info()
        
        # Validate action is to a real node or depot
        if action >= self.original_n_nodes:
            return self._get_observation(), -1000.0, True, False, self._get_info()
        
        if not self._is_action_valid(action):
            return self._get_observation(), -1000.0, True, False, self._get_info()
        
        # Update time and reveal customers
        self.time_step += 1
        self._update_revealed_customers()
        
        # Execute action
        reward = self._execute_action(action)
        
        # Check termination
        terminated = self._is_terminated()
        truncated = (self.time_step >= 500)
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _update_revealed_customers(self):
        """Update revealed customers based on time"""
        for customer_id, arrival_time in self.arrival_times.items():
            if self.time_step >= arrival_time:
                self.revealed[customer_id] = True
    
    def _execute_action(self, action: int) -> float:
        """Execute the action"""
        prev_location = self.current_location
        
        # Calculate distance using original coordinates
        distance = float(self.distance_matrix[prev_location, action])
        self.total_distance += distance
        
        # Move to new location
        self.current_location = action
        self.current_route.append(action)
        
        if action == 0:  # Depot
            # Complete route if it has customers
            if len(self.current_route) > 2:
                self.completed_routes.append(self.current_route.copy())
                self.n_vehicles_used += 1
            
            # Reset for new route
            self.current_route = [0]
            self.current_capacity = self.capacity
        else:  # Customer
            # Serve customer
            self.visited[action] = True
            self.current_capacity -= self.demands[action]
        
        if self._is_terminated():
            total_objective = self.total_distance + self.vehicle_cost * self.n_vehicles_used
            return -total_objective
        else:
            return -distance  # Step-
    
    def _is_action_valid(self, action: int) -> bool:
        """Check if action is valid"""
        # Can't go to padding positions
        if action >= self.original_n_nodes:
            return False
        
        if action == 0:  # Depot
            if self.current_location != 0:
                return True  # Can return to depot
            else:
                # At depot - only valid if no serviceable customers
                serviceable = self._get_serviceable_customers()
                return len(serviceable) == 0
        else:  # Customer
            return action in self._get_serviceable_customers()
    
    def _get_serviceable_customers(self) -> List[int]:
        """Get serviceable customers (only real customers)"""
        serviceable = []
        for i in range(1, self.original_n_nodes):  # Only real customers
            if (self.revealed[i] and 
                not self.visited[i] and 
                self.demands[i] <= self.current_capacity):
                serviceable.append(i)
        return serviceable
    
    def _is_terminated(self) -> bool:
        """Check if episode is terminated"""
        # All revealed real customers served?
        unserved = [i for i in range(1, self.original_n_nodes) 
                   if self.revealed[i] and not self.visited[i]]
        
        if len(unserved) == 0:
            # Check for future arrivals (only for real customers)
            future_arrivals = any(
                customer_id < self.original_n_nodes and arrival_time > self.time_step 
                for customer_id, arrival_time in self.arrival_times.items()
            )
            
            if not future_arrivals:
                # Force return to depot
                if self.current_location != 0:
                    final_distance = float(self.distance_matrix[self.current_location, 0])
                    self.total_distance += final_distance
                    self.current_route.append(0)
                
                # Complete final route
                if len(self.current_route) > 2:
                    self.completed_routes.append(self.current_route.copy())
                    self.n_vehicles_used += 1
                
                return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Get observation with original coordinates"""
        obs = []
        
        # Depot coordinates (original scale)
        obs.extend(self.coords[0])
        
        # Customer features (including padding)
        for i in range(1, self.n_nodes):
            if i < self.original_n_nodes:  # Real customer
                # Absolute coordinates (not relative)
                obs.extend([
                    float(self.coords[i, 0]),
                    float(self.coords[i, 1]),
                    float(self.demands[i]) / np.max(self.original_demands[1:]) if np.max(self.original_demands[1:]) > 0 else 0.0,
                    1.0 if self.revealed[i] else 0.0
                ])
            else:  # Padding customer
                # Padding positions: depot coordinates, zero demand, never revealed
                obs.extend([
                    float(self.coords[0, 0]),  # Depot x
                    float(self.coords[0, 1]),  # Depot y
                    0.0,  # Zero demand
                    0.0   # Never revealed
                ])
        
        # Vehicle state
        current_node_norm = float(self.current_location) / max(1, self.original_n_nodes - 1)
        capacity_norm = self.current_capacity / self.capacity
        time_norm = min(self.time_step / 100.0, 1.0)
        
        obs.extend([current_node_norm, capacity_norm, time_norm])
        
        return np.array(obs, dtype=np.float32)
    
    def action_masks(self) -> np.ndarray:
        """Get action mask"""
        mask = np.zeros(self.n_nodes, dtype=bool)
        
        # Only allow actions to real nodes
        for action in range(self.original_n_nodes):
            mask[action] = self._is_action_valid(action)
        
        # Never allow actions to padding positions
        for action in range(self.original_n_nodes, self.n_nodes):
            mask[action] = False
        
        # Ensure at least depot is valid
        if not mask.any():
            mask[0] = True
        
        return mask
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary with original distances"""
        served_customers = int(np.sum(self.visited[1:self.original_n_nodes]))
        revealed_customers = int(np.sum(self.revealed[1:self.original_n_nodes]))
        total_customers = self.original_n_customers
        
        objective_cost = self.total_distance + self.vehicle_cost * self.n_vehicles_used
        
        return {
            'total_distance': self.total_distance,
            'n_vehicles_used': self.n_vehicles_used,
            'objective_cost': objective_cost,
            'customers_served': served_customers,
            'customers_revealed': revealed_customers,
            'total_customers': total_customers,
            'completion_rate': served_customers / max(total_customers, 1),
            'time_step': self.time_step,
            'current_location': self.current_location,
            'remaining_capacity': self.current_capacity,
            'action_mask': self.action_masks(),
            'finished': served_customers == total_customers,
            'all_routes': [route.copy() for route in self.completed_routes],
            'current_route': self.current_route.copy()
        }
    
    def render(self, mode: str = 'human'):
        """Render environment"""
        if mode == 'human':
            info = self._get_info()
            print(f"Step {self.time_step}: At {info['current_location']}, "
                  f"Capacity {info['remaining_capacity']:.1f}/{self.capacity}, "
                  f"Served {info['customers_served']}/{info['total_customers']}")
    
    def close(self):
        """Clean up"""
        pass

def create_env(coords, demands, capacity, num_vehicles, seed=None, vehicle_cost=50.0, max_customers=100):
    """Create environment with existing interface"""
    return SCVRPEnv(
        coords=coords,
        demands=demands,
        capacity=capacity,
        vehicle_cost=vehicle_cost,
        max_customers=max_customers,
        seed=seed
    )