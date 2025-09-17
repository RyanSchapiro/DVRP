import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Maximum number of customers supported in observation space - creates fixed size regardless of problem
MAX_CUSTOMERS = 100

class VRP(gym.Env):
    """Gymnasium environment for Dynamic Vehicle Routing Problem (VRP)."""
    
    def __init__(self, coords, demands, capacity, max_vehicles=None, seed=42, static_ratio=0.7, arrival_rate=0.4):
        """Initialize VRP environment.
        
        Args:
            coords: numpy array of (x,y) coordinates [depot, customer1, customer2, ...]
            demands: numpy array of demands [0, demand1, demand2, ...] (depot has 0 demand)
            capacity: Vehicle capacity constraint
            max_vehicles: Maximum number of vehicles allowed (auto-calculated if None)
            seed: Random seed for reproducible customer arrivals
            static_ratio: Fraction of customers known at start (0.0-1.0)
            arrival_rate: Poisson rate parameter for dynamic arrivals
        """
        super().__init__()
        
        # Store problem instance data
        self.coords = coords.copy() 
        self.demands = demands.copy()
        self.capacity = float(capacity)
        self.n_customers = len(demands) - 1  # Exclude depot
        self.seed = seed
        
        self.static_ratio = static_ratio 
        self.arrival_rate = arrival_rate 
        
        # Calculate reasonable upper bound on vehicles needed
        if max_vehicles is None:
            total_demand = np.sum(demands[1:])
            self.max_vehicles = max(1, int(np.ceil(total_demand / capacity)) + 2)
        else:
            self.max_vehicles = max_vehicles

        # Precompute distance matrix for efficiency
        self.dist_matrix = self._compute_distances()
        
        # Action Space: Choose which customer to visit next (or return to depot)
        # Action 0 = return to depot, Action i = visit customer i
        self.action_space = spaces.Discrete(self.n_customers + 1)

        # Observation Space: Fixed-size vector regardless of problem size
        # Structure: [current_pos_x, current_pos_y, remaining_capacity, time] + 
        #           [for each possible customer: x, y, demand, revealed, visited]
        self.max_customers_universal = MAX_CUSTOMERS
        obs_dim = 4 + 5 * MAX_CUSTOMERS  # 4 global features + 5 features per customer
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        
        # Normalization constants for observations
        self.max_coord = np.max(np.abs(coords))
        self.max_demand = np.max(demands) if len(demands) > 1 else 1
        
        self.reset()
    
    def _compute_distances(self):
        """Compute Euclidean distance matrix between all locations.
        
        Returns:
            Integer distance matrix where dist_matrix[i][j] = distance from i to j
        """
        distances = np.linalg.norm(self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :], axis=-1)
        return distances.astype(int)
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state for new episode.
        
        Args:
            seed: Optional seed for this episode
            options: Additional reset options (unused)
            
        Returns:
            Tuple of (initial_observation, info_dict)
        """
        if seed is not None:
            self.seed = seed
        np.random.seed(self.seed)
        
        # Generate which customers are static vs dynamic
        self._generate_requests()

        # Initialize episode state
        self.current_pos = 0                                    # Start at depot
        self.remaining_capacity = self.capacity                 # Full capacity
        self.visited = np.zeros(len(self.demands), dtype=bool)  # Track visited customers
        self.visited[0] = True                                  # Depot always "visited"
        self.time = 0                                          # Current time step
        
        # Solution tracking
        self.total_distance = 0.0           # Total distance traveled
        self.current_route = [0]            # Current route (starts at depot)
        self.all_routes = []                # Completed routes
        self.n_vehicles_used = 0            # Number of vehicles used
        
        return self._get_obs(), self._get_info()
    
    def _generate_requests(self):
        """Generate static and dynamic customer arrival schedule."""

        from data import Poisson

        static_customers, dynamic_arrivals = Poisson(
            dim=len(self.demands),
            stat=self.static_ratio,
            lam=self.arrival_rate,
            seed=self.seed
        )

        # Track which customers are currently revealed to the agent
        self.revealed = np.zeros(len(self.demands), dtype=bool)
        self.revealed[0] = True  # Depot always known
        
        # Reveal static customers immediately
        for customer in static_customers:
            if customer < len(self.demands):
                self.revealed[customer] = True
        
        # Store dynamic arrival times
        self.arrival_times = dynamic_arrivals
    
    def step(self, action):
        """Execute one step in the environment.
        
        Args:
            action: Integer action (0=depot, 1+=customer index)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
            - observation: Current state representation
            - reward: Negative distance traveled (minimize cost)
            - terminated: True if episode complete
            - truncated: False (not used)
            - info: Dictionary with detailed episode information
        """
        self.time += 1
        
        # Check for new customer arrivals at current time
        self._reveal_customers()
        
        # Validate and potentially correct invalid actions
        if action >= len(self.demands):
            action = 0
        
        if not self._is_valid_action(action):
            # Try returning to depot as fallback
            if self._is_valid_action(0):
                action = 0
            else:
                # No valid actions available - terminate with penalty
                return self._get_obs(), -1000, True, False, self._get_info()
        
        # Execute action: move from current position to chosen destination
        step_distance = self.dist_matrix[self.current_pos, action]
        self.total_distance += step_distance
        
        self.current_pos = action
        self.current_route.append(action)
        
        # Reward is negative distance
        reward = -step_distance
        
        if action == 0:
            # Returned to depot - complete current route
            route_has_customers = any(node != 0 for node in self.current_route[1:-1])
            
            if route_has_customers:
                self.all_routes.append(self.current_route.copy())
            
            # Start new route
            self.current_route = [0]
            self.remaining_capacity = self.capacity
            
        else:
            # Visited customer - update state
            self.visited[action] = True
            self.remaining_capacity -= self.demands[action]
        
        # Check termination conditions
        unvisited_revealed = self._get_unvisited_revealed()
        
        if not np.any(unvisited_revealed):
            # No more revealed customers to visit
            # Check if any customers will arrive in the future
            future_arrivals = any(
                arrival_time > self.time 
                for arrival_time in self.arrival_times.values()
            )
            
            if not future_arrivals:
                # All customers processed - episode complete
                if self.current_pos != 0:
                    # Return to depot if not already there
                    final_distance = self.dist_matrix[self.current_pos, 0]
                    self.total_distance += final_distance
                    reward -= final_distance
                    self.current_route.append(0)
                
                # Save final route if it has customers
                final_route_has_customers = any(node != 0 for node in self.current_route[1:-1])
                if final_route_has_customers:
                    self.all_routes.append(self.current_route.copy())

                terminated = True
            else:
                terminated = False
        else:
            terminated = False
        
        # Safety check: if no valid actions available, terminate
        valid_actions = self._get_action_mask()
        if not np.any(valid_actions) and not terminated:
            reward -= 100  # Penalty for getting stuck
            terminated = True
        
        return self._get_obs(), reward, terminated, False, self._get_info()
    
    def _reveal_customers(self):
        """Reveal customers whose arrival time has come."""
        for customer, arrival_time in self.arrival_times.items():
            if self.time >= arrival_time and not self.revealed[customer]:
                self.revealed[customer] = True
    
    def _get_unvisited_revealed(self):
        """Get boolean array of customers that are revealed but not yet visited.
        
        Returns:
            Boolean array where True indicates customer is revealed and unvisited
        """
        unvisited_revealed = self.revealed & ~self.visited
        unvisited_revealed[0] = False  # Depot doesn't count
        return unvisited_revealed
    
    def _get_obs(self):
        """Construct observation vector from current environment state.
        
        Observation structure:
        - [0-3]: Global state (current position x,y, remaining capacity, time)  
        - [4+]: Customer features (x, y, demand, revealed, visited) for each customer
        
        All values normalized to [0,1] range for stable learning.
        
        Returns:
            Normalized observation vector as numpy array
        """
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Global state features
        obs[0] = self.coords[self.current_pos, 0] / self.max_coord  # Current x position
        obs[1] = self.coords[self.current_pos, 1] / self.max_coord  # Current y position
        obs[2] = self.remaining_capacity / self.capacity            # Remaining capacity
        obs[3] = min(self.time / 100.0, 1.0)                      # Time (capped at 100)
        
        # Customer features (fixed-size array regardless of actual problem size)
        for i in range(MAX_CUSTOMERS):
            base_idx = 4 + i * 5
            
            if i < self.n_customers:
                customer_idx = i + 1  # Skip depot (index 0)
                obs[base_idx] = self.coords[customer_idx, 0] / self.max_coord      # x coordinate
                obs[base_idx + 1] = self.coords[customer_idx, 1] / self.max_coord  # y coordinate  
                obs[base_idx + 2] = self.demands[customer_idx] / self.max_demand   # demand
                obs[base_idx + 3] = 1.0 if self.revealed[customer_idx] else 0.0   # revealed flag
                obs[base_idx + 4] = 1.0 if self.visited[customer_idx] else 0.0    # visited flag
            else:
                # Padding for unused customer slots (all zeros)
                pass
        
        return obs
    
    def _is_valid_action(self, action):
        """Check if given action is valid in current state.
        
        Args:
            action: Action to validate
            
        Returns:
            True if action is valid, False otherwise
            
        Action is valid if:
        - Action 0 (depot) is always valid
        - Customer actions are valid if: revealed, not visited, demand fits in capacity
        """
        if action == 0:
            return True  # Can always return to depot
        
        if action >= len(self.demands):
            return False  # Invalid customer index
        
        return (self.revealed[action] and 
                not self.visited[action] and 
                self.demands[action] <= self.remaining_capacity)
    
    def _get_action_mask(self):
        """Get boolean mask indicating which actions are currently valid.
        
        Used by some RL algorithms to avoid selecting invalid actions.
        
        Returns:
            Boolean array where True indicates valid action
        """
        mask = np.zeros(MAX_CUSTOMERS + 1, dtype=bool)
        
        mask[0] = True  # Depot always valid
        
        # Check validity of each customer action
        for i in range(min(self.n_customers, MAX_CUSTOMERS)):
            customer_idx = i + 1
            if customer_idx < len(self.demands):
                mask[i + 1] = self._is_valid_action(customer_idx)
        
        return mask

    def _get_info(self):
        """Construct detailed information dictionary about current episode state.
        
        Returns:
            Dictionary with comprehensive episode statistics and state information.
            Used for monitoring, debugging, and analysis.
        """
        unvisited_revealed = self._get_unvisited_revealed()
        total_unvisited = np.sum(~self.visited[1:])      # Total customers not yet visited
        revealed_unvisited = np.sum(unvisited_revealed)   # Revealed customers not yet visited

        # Calculate number of vehicles used
        completed_vehicles = len(self.all_routes)
        current_has_customers = any(node != 0 for node in self.current_route[1:])
        total_vehicles = completed_vehicles + (1 if current_has_customers else 0)
        
        # Calculate full objective including vehicle fixed costs
        vehicle_cost = 50.0
        full_objective = self.total_distance + vehicle_cost * total_vehicles
        
        return {
            'total_distance': self.total_distance,              # Travel distance
            'n_vehicles_used': total_vehicles,                  # Number of vehicles
            'customers_served': self.n_customers - total_unvisited,  # Progress
            'customers_remaining': total_unvisited,             # Remaining work
            'revealed_remaining': revealed_unvisited,           # Immediate work available
            'remaining_capacity': self.remaining_capacity,      # Current vehicle capacity
            'current_route': self.current_route.copy(),         # Current route
            'all_routes': [route.copy() for route in self.all_routes],  # All completed routes
            'action_mask': self._get_action_mask(),            # Valid actions
            'completion_rate': (self.n_customers - total_unvisited) / self.n_customers,  # Progress %
            'time': self.time,                                 # Current time step
            'revealed': self.revealed.copy(),                   # Customer revelation status
            'arrival_times': self.arrival_times.copy(),        # Arrival schedule
            'objective_cost': full_objective,                  # Total cost (distance + vehicles)
            'complete': total_unvisited == 0                   # Episode completion flag
        }