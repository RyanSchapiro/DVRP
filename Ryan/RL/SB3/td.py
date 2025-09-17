import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import tensorboard
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import ProgressBarCallback
from gymnasium import spaces
import gymnasium as gym
import math
from data import load_data

class GATLayer(nn.Module):
    """Graph Attention Network layer for processing customer-to-customer relationships.
    
    Uses multi-head attention to let each customer attend to all other customers,
    learning which customers are most relevant for routing decisions. This captures
    spatial relationships and routing patterns better than treating customers independently.
    """
    
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        """Initialize GAT layer.
        
        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension  
            num_heads: Number of attention heads for multi-head attention
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads
        
        # Multi-head attention mechanism - core of graph attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Batch dimension first for easier processing
        )
        
        # Project attention output to desired dimension
        self.out_proj = nn.Linear(in_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)  # Stabilize training
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, node_mask):
        """Forward pass through GAT layer.
        
        Args:
            x: Input node features [batch_size, n_nodes, in_dim]
            node_mask: Boolean mask indicating valid nodes [batch_size, n_nodes]
            
        Returns:
            Updated node features after attention [batch_size, n_nodes, out_dim]
        """
        batch_size, n_nodes, in_dim = x.shape
        
        # Apply self-attention: each customer attends to all other customers
        # key_padding_mask prevents attention to invalid/non-existent customers
        attn_output, _ = self.multihead_attn(
            query=x,  # What each customer is looking for
            key=x,    # What each customer offers as context
            value=x,  # The actual information to aggregate
            key_padding_mask=~node_mask  # Mask invalid customers
        )
        
        # Residual connection + layer normalization (standard transformer pattern)
        residual = self.out_proj(attn_output)
        output = self.layer_norm(residual + self.out_proj(x))
        output = self.dropout(output)
        
        return output

class GATExtractor(BaseFeaturesExtractor):
    """Custom feature extractor using Graph Attention Networks.
    
    This replaces the standard MLP feature extractor in RL algorithms with a GAT-based
    architecture that better understands spatial relationships between customers.
    
    Architecture:
    1. Embed individual customer features and global state features
    2. Apply GAT layers to learn customer interactions
    3. Aggregate customer information with attention-based pooling
    4. Combine with global state to produce final feature representation
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        """Initialize GAT-based feature extractor.
        
        Args:
            observation_space: Gym observation space (determines input size)
            features_dim: Output feature dimension for policy/value networks
        """
        # Parse observation structure from VRP environment
        total_obs_size = observation_space.shape[0]
        self.state_features = 4  # Global features: pos_x, pos_y, capacity, time
        self.customer_feature_size = 5  # Per-customer: x, y, demand, revealed, visited
        self.max_customers = (total_obs_size - self.state_features) // self.customer_feature_size
        
        super().__init__(observation_space, features_dim)
        
        self.embed_dim = 64  # Embedding dimension for both customers and state
        
        # Embedding layers to project raw features to consistent dimension
        self.customer_embedder = nn.Linear(self.customer_feature_size, self.embed_dim)
        self.state_embedder = nn.Linear(self.state_features, self.embed_dim)
        
        # Two GAT layers for learning customer interactions
        # Multiple layers allow learning higher-order relationships
        self.gat1 = GATLayer(self.embed_dim, self.embed_dim, num_heads=4, dropout=0.1)
        self.gat2 = GATLayer(self.embed_dim, self.embed_dim, num_heads=4, dropout=0.1)
        
        # Final MLP to combine customer context with global state
        self.final_layer = nn.Sequential(
            nn.Linear(self.embed_dim * 2, features_dim),  # Customer context + state context
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim)
        )
        
        # Initialize weights for stable training
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize neural network weights for stable training."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.1)  # Small initial weights
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features from VRP observations using GAT architecture.
        
        Args:
            observations: Batch of VRP observations [batch_size, obs_dim]
            
        Returns:
            Processed features for policy network [batch_size, features_dim]
        """
        batch_size = observations.shape[0]
        
        # Split observation into global state and customer features
        state_features = observations[:, :self.state_features]
        customer_features = observations[:, self.state_features:]
        customer_features = customer_features.view(batch_size, self.max_customers, self.customer_feature_size)
        
        # Handle NaN values that might occur during training
        if torch.isnan(observations).any():
            observations = torch.nan_to_num(observations, nan=0.0)
        
        # Embed customer features to consistent dimension
        customer_embeddings = self.customer_embedder(customer_features)
        
        # Create mask for valid customers (revealed flag > 0.5)
        # This tells GAT which customers actually exist vs padding
        customer_exists = customer_features[:, :, 3] > 0.5
        
        # Ensure at least one customer exists per batch (safety check)
        for b in range(batch_size):
            if not customer_exists[b].any():
                customer_exists[b, 0] = True
        
        # Apply GAT layers to learn customer interactions
        h = customer_embeddings
        h = self.gat1(h, customer_exists)  # First GAT layer
        h = self.gat2(h, customer_exists)  # Second GAT layer
        
        # Additional NaN safety check after GAT processing
        if torch.isnan(h).any():
            h = torch.nan_to_num(h, nan=0.0)
        
        # Aggregate customer information using masked average pooling
        # Only pool over valid customers, ignore padding
        mask_expanded = customer_exists.unsqueeze(-1).float()
        valid_count = mask_expanded.sum(dim=1, keepdim=True).clamp(min=1.0)
        customer_context = (h * mask_expanded).sum(dim=1) / valid_count.squeeze(-1)
        
        # Embed global state features
        state_context = self.state_embedder(state_features)
        
        # Combine customer context with global state
        combined = torch.cat([customer_context, state_context], dim=-1)
        output = self.final_layer(combined)
        
        # Final safety checks for numerical stability
        if torch.isnan(output).any():
            output = torch.nan_to_num(output, nan=0.0)
        
        output = torch.clamp(output, -10.0, 10.0)  # Prevent extreme values
        
        return output

class Wrap(gym.Wrapper):
    """Wrapper to convert discrete VRP actions to continuous actions for SAC algorithm.
    
    SAC requires continuous action spaces, but VRP naturally has discrete actions
    (visit customer X). This wrapper maps continuous actions to discrete customer
    selections using the savings heuristic.

    """
    
    def __init__(self, env):
        """Initialize action space wrapper.
        
        Args:
            env: Base VRP environment with discrete actions
        """
        super().__init__(env)
        # Convert to continuous action space for SAC
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self._discrete_action_space = env.action_space
        
        # Exploration parameters for action selection
        self.exploration_low = 0.3   # Below this: greedy selection
        self.exploration_high = 0.7  # Above this: more random selection
        self.noise_std = 0.1         # Action noise for exploration
    
    def reset(self, **kwargs):
        """Reset environment and store action mask."""
        obs, info = self.env.reset(**kwargs)
        self._action_mask = info['action_mask']
        return obs, info
    
    def step(self, action):
        """Convert continuous action to discrete action and execute.
        
        Args:
            action: Continuous action from SAC agent [-1, 1]
            
        Returns:
            Standard gym step return tuple
        """
        # Extract scalar action value
        if isinstance(action, np.ndarray):
            action_value = action[0]
        else:
            action_value = float(action)
        
        # Get valid actions from environment
        valid_indices = np.where(self._action_mask)[0]
        
        if len(valid_indices) == 0:
            discrete_action = 0  # Return to depot as fallback
        else:
            # Use intelligent heuristic to select discrete action
            discrete_action = self.select(action_value, valid_indices)
        
        # Execute discrete action in environment
        obs, reward, terminated, truncated, info = self.env.step(discrete_action)
        self._action_mask = info['action_mask']
        
        return obs, reward, terminated, truncated, info
        
    def select(self, action_value, valid_indices):
        """Intelligent action selection heuristic
        
        Args:
            action_value: Continuous action value [-1, 1] from SAC
            valid_indices: Valid discrete actions available
            
        Returns:
            Selected discrete action (customer index or depot)
        """
        depot_available = 0 in valid_indices
        customer_indices = valid_indices[valid_indices > 0]
        
        # If no customers available, go to depot
        if len(customer_indices) == 0:
            return 0 if depot_available else valid_indices[0]
        
        # Filter customers by capacity constraint
        serveable_customers = []
        for customer in customer_indices:
            if self.env.demands[customer] <= self.env.remaining_capacity:
                serveable_customers.append(customer)
        
        # If no customers fit in current vehicle, return to depot
        if len(serveable_customers) == 0:
            return 0 if depot_available else customer_indices[0]
        
        current_pos = self.env.current_pos
        
        # Heuristic: Return to depot if it's closer and vehicle nearly full
        if depot_available and current_pos != 0:
            depot_distance = self.env.dist_matrix[current_pos, 0]
            min_customer_distance = min([self.env.dist_matrix[current_pos, c] for c in serveable_customers])
            
            capacity_used = 1.0 - (self.env.remaining_capacity / self.env.capacity)
            if depot_distance < min_customer_distance and capacity_used > 0.95:
                return 0
        
        # Calculate customer attractiveness using VRP heuristics
        if current_pos == 0:
            # At depot: use distance-based weights (closer customers preferred)
            distances = np.array([self.env.dist_matrix[0, c] for c in serveable_customers])
            weights = 1.0 / (distances + 1e-6)  # Inverse distance weighting
        else:
            # In route: use savings heuristic (Clarke-Wright inspired)
            savings = []
            for customer in serveable_customers:
                direct_cost = self.env.dist_matrix[current_pos, customer]
                detour_cost = self.env.dist_matrix[current_pos, 0] + self.env.dist_matrix[0, customer]
                saving = detour_cost - direct_cost  # Distance saved by visiting directly
                savings.append(max(saving, 0.1))   # Ensure positive weights
            
            weights = np.array(savings)
        
        # Normalize weights to probabilities
        weights = weights / np.sum(weights)
        
        # Convert continuous action to exploration level [0, 1]
        exploration = (action_value + 1) / 2  # Map [-1, 1] to [0, 1]
        exploration = np.clip(exploration, 0, 1)
        
        # Use exploration level to control selection strategy
        if exploration < 0.3:
            # Low exploration: greedy selection (best customer)
            return serveable_customers[np.argmax(weights)]
        else:
            # High exploration: weighted random selection
            return serveable_customers[np.random.choice(len(serveable_customers), p=weights)]

def create_model(logdir, algorithm='SAC', env=None, **kwargs):
    """Create RL model
    
    Args:
        logdir: Directory for tensorboard logs
        algorithm: RL algorithm to use (currently only SAC supported)
        env: Training environment
        **kwargs: Additional model arguments
        
    Returns:
        Configured RL model
    """
    # Configure policy network with custom GAT feature extractor
    policy_kwargs = {
        'features_extractor_class': GATExtractor,      # Use custom GAT extractor
        'features_extractor_kwargs': {'features_dim': 128},
        'net_arch': [64, 64],                         # MLP layers after feature extraction
        'activation_fn': torch.nn.ReLU,               # Activation function
    }

    tensorboard_log = logdir
    
    if algorithm == 'SAC':
        from stable_baselines3 import SAC
        
        # SAC hyperparameters tuned for VRP
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=1e-4,        # Learning rate
            buffer_size=100000,        # Replay buffer size
            learning_starts=10000,     # Steps before training starts
            batch_size=256,            # Mini-batch size
            tau=0.005,                # Soft update coefficient
            gamma=0.95,               # Discount factor
            train_freq=4,             # Training frequency
            gradient_steps=1,         # Gradient steps per update
            ent_coef="auto_0.1",      # Entropy coefficient (automatic tuning)
            target_entropy="auto",    # Target entropy for automatic tuning
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=1,
            **kwargs
        )
    
    return model

def create_env(coords, demands, capacity, num_vehicles, seed=None, vehicle_cost=50.0, max_customers=100):
    """Create wrapped VRP environment for RL training.
    
    Args:
        coords: Customer coordinates
        demands: Customer demands
        capacity: Vehicle capacity
        num_vehicles: Number of vehicles available
        seed: Random seed
        vehicle_cost: Fixed cost per vehicle
        max_customers: Maximum customers supported
        
    Returns:
        Wrapped and monitored VRP environment
    """
    from dvrp_env import VRP
    from stable_baselines3.common.monitor import Monitor
    
    # Calculate reasonable vehicle limit based on problem size
    max_vehicles = max(3, min(15, len(demands) // 2))
    
    # Create base VRP environment
    env = VRP(
        coords=coords,
        demands=demands, 
        capacity=capacity,
        max_vehicles=max_vehicles,
        seed=np.random.randint(0, 10000)  # Random seed for each episode
    )
    
    # Apply wrappers for RL compatibility
    env = Wrap(env)      # Convert discrete to continuous actions
    env = Monitor(env)   # Monitor training statistics
    
    return env

def train_model(instance, total_steps=200000, vehicle_cost=50.0, max_customers=100, testfile=""):
    """Train RL model on VRP instance.
    
    Args:
        instance: Instance identifier for logging
        total_steps: Total training timesteps
        vehicle_cost: Fixed cost per vehicle
        max_customers: Maximum customers supported
        testfile: Path to problem instance file
        
    Returns:
        Trained RL model
    """
    from generate import generate_instance
    
    # Load problem instance
    coords, demands, capacity, vehichle_count = load_data(testfile)
    
    # Create training environment
    env = create_env(coords, demands, capacity, 25, 
                     vehicle_cost=vehicle_cost, max_customers=max_customers)
    
    # Setup logging
    logdir = f"./logs/vrp/" + instance
    
    # Create model with custom architecture
    model = create_model(logdir, 'SAC', env)
    
    # Setup training callbacks
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5, min_evals=10, verbose=1
    )

    eval_callback = EvalCallback(
        env, eval_freq=2000, callback_after_eval=stop_callback, verbose=1
    )

    # Train the model
    model.learn(
        total_timesteps=total_steps, 
        callback=[ProgressBarCallback(), eval_callback]
    )
    
    env.close()
    return model

def main():
    algorithm = "SAC"
    
    # Get training configuration from user
    test_file = "../../dataset/files/ " + input("Test file [c1_25.txt]:").strip() or "c1_25.txt"
    
    instance = input("Instance type (R/C/RC) [R]: ").strip().upper() or "R"

    steps_input = input("Total training steps [200000]: ").strip()
    total_steps = int(steps_input) if steps_input else 200000
    
    cost_input = input("Vehicle cost [50]: ").strip()
    vehicle_cost = float(cost_input) if cost_input else 50.0
    
    customer = (input("Max customers [100]: ").strip())
    customers = int(customer) if customer else 100
    
    # Train the model
    model = train_model(
        instance=instance,
        total_steps=total_steps,
        vehicle_cost=vehicle_cost,
        max_customers=customers,
        testfile=test_file
    )
    
    # Save trained model
    model_path = f"{customers}_{total_steps//1000}"
    model.save(model_path)

if __name__ == "__main__":
    main()