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



class StableGATLayer(nn.Module):
    """Numerically stable Graph Attention Layer"""
    
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads
        
        # Multi-head attention (much more stable)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.out_proj = nn.Linear(in_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, node_mask):
        """
        x: [batch, n_nodes, in_dim]
        node_mask: [batch, n_nodes] - True for valid nodes
        """
        batch_size, n_nodes, in_dim = x.shape
        
        # Use PyTorch's stable multihead attention
        attn_output, _ = self.multihead_attn(
            query=x,
            key=x, 
            value=x,
            key_padding_mask=~node_mask  # False for valid nodes
        )
        
        # Residual connection + layer norm
        residual = self.out_proj(attn_output)
        output = self.layer_norm(residual + self.out_proj(x))
        output = self.dropout(output)
        
        return output


class SimpleGATExtractor(BaseFeaturesExtractor):
    """Stable GAT feature extractor - direct replacement"""
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        # Same parsing as your original
        total_obs_size = observation_space.shape[0]
        self.state_features = 4
        self.customer_feature_size = 5
        self.max_customers = (total_obs_size - self.state_features) // self.customer_feature_size
        
        super().__init__(observation_space, features_dim)
        
        self.embed_dim = 64
        
        # Same embedders
        self.customer_embedder = nn.Linear(self.customer_feature_size, self.embed_dim)
        self.state_embedder = nn.Linear(self.state_features, self.embed_dim)
        
        # Stable GAT layers
        self.gat1 = StableGATLayer(self.embed_dim, self.embed_dim, num_heads=4, dropout=0.1)
        self.gat2 = StableGATLayer(self.embed_dim, self.embed_dim, num_heads=4, dropout=0.1)
        
        # Same final layer with gradient clipping
        self.final_layer = nn.Sequential(
            nn.Linear(self.embed_dim * 2, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim)
        )
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Proper weight initialization to prevent NaNs"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.1)  # Small gain
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # Same parsing
        state_features = observations[:, :self.state_features]
        customer_features = observations[:, self.state_features:]
        customer_features = customer_features.view(batch_size, self.max_customers, self.customer_feature_size)
        
        # Check for NaN inputs
        if torch.isnan(observations).any():
            print("WARNING: NaN in input observations!")
            observations = torch.nan_to_num(observations, nan=0.0)
        
        # Embed customers
        customer_embeddings = self.customer_embedder(customer_features)
        customer_exists = customer_features[:, :, 3] > 0.5
        
        # Ensure at least one customer exists per batch
        for b in range(batch_size):
            if not customer_exists[b].any():
                customer_exists[b, 0] = True  # Force first customer to exist
        
        # Apply GAT layers
        h = customer_embeddings
        h = self.gat1(h, customer_exists)
        h = self.gat2(h, customer_exists)
        
        # Check for NaN in GAT outputs
        if torch.isnan(h).any():
            print("WARNING: NaN in GAT output!")
            h = torch.nan_to_num(h, nan=0.0)
        
        # Global pooling with numerical stability
        mask_expanded = customer_exists.unsqueeze(-1).float()
        valid_count = mask_expanded.sum(dim=1, keepdim=True).clamp(min=1.0)  # Prevent division by 0
        customer_context = (h * mask_expanded).sum(dim=1) / valid_count.squeeze(-1)
        
        # State context
        state_context = self.state_embedder(state_features)
        
        # Combine with gradient clipping
        combined = torch.cat([customer_context, state_context], dim=-1)
        
        # Apply final layer with gradient clipping
        output = self.final_layer(combined)
        
        # Final NaN check
        if torch.isnan(output).any():
            print("WARNING: NaN in final output!")
            output = torch.nan_to_num(output, nan=0.0)
        
        # Clamp output to reasonable range
        output = torch.clamp(output, -10.0, 10.0)
        
        return output



class Wrap(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self._discrete_action_space = env.action_space
        
        # Add missing attributes
        self.exploration_low = 0.3
        self.exploration_high = 0.7
        self.noise_std = 0.1
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._action_mask = info['action_mask']
        return obs, info
    
    def step(self, action):
        # Convert continuous action to discrete
        if isinstance(action, np.ndarray):
            action_value = action[0]
        else:
            action_value = float(action)
        
        # Get valid actions
        valid_indices = np.where(self._action_mask)[0]
        
        if len(valid_indices) == 0:
            discrete_action = 0  # Fallback to depot
        else:
            discrete_action = self.select(action_value, valid_indices)
        
        # Execute action
        obs, reward, terminated, truncated, info = self.env.step(discrete_action)
        self._action_mask = info['action_mask']
        
        return obs, reward, terminated, truncated, info
    
    
        
    def select(self, action_value, valid_indices):
        depot_available = 0 in valid_indices
        customer_indices = valid_indices[valid_indices > 0]
        
        if len(customer_indices) == 0:
            return 0 if depot_available else valid_indices[0]
        
        # Filter by capacity
        serveable_customers = []
        for customer in customer_indices:
            if self.env.demands[customer] <= self.env.remaining_capacity:
                serveable_customers.append(customer)
        
        if len(serveable_customers) == 0:
            return 0 if depot_available else customer_indices[0]
        
        current_pos = self.env.current_pos
        
        # Fixed savings calculation - only apply when NOT at depot
        if depot_available and current_pos != 0:  # Only consider depot return if not already there
            # Calculate route completion cost vs extension cost
            depot_distance = self.env.dist_matrix[current_pos, 0]
            min_customer_distance = min([self.env.dist_matrix[current_pos, c] for c in serveable_customers])
            
            # Return to depot if it's closer than continuing AND capacity > 80%
            capacity_used = 1.0 - (self.env.remaining_capacity / self.env.capacity)
            if depot_distance < min_customer_distance and capacity_used > 0.95:
                return 0
        
        # Customer selection using proper savings
        if current_pos == 0:
            # At depot - use distance-based selection
            distances = np.array([self.env.dist_matrix[0, c] for c in serveable_customers])
            weights = 1.0 / (distances + 1e-6)
        else:
            # Not at depot - use savings for route extension
            savings = []
            for customer in serveable_customers:
                # Saving = how much we save by visiting customer now vs returning to depot then visiting
                direct_cost = self.env.dist_matrix[current_pos, customer]
                detour_cost = self.env.dist_matrix[current_pos, 0] + self.env.dist_matrix[0, customer]
                saving = detour_cost - direct_cost
                savings.append(max(saving, 0.1))  # Minimum positive weight
            
            weights = np.array(savings)
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Apply exploration
        exploration = (action_value + 1) / 2
        exploration = np.clip(exploration, 0, 1)
        
        if exploration < 0.3:
            # Exploitation - choose best option
            return serveable_customers[np.argmax(weights)]
        else:
            # Exploration - weighted random
            return serveable_customers[np.random.choice(len(serveable_customers), p=weights)]
        


def create_model(logdir, algorithm='SAC', env=None, **kwargs):

    
    policy_kwargs = {
        'features_extractor_class': SimpleGATExtractor,
        'features_extractor_kwargs': {'features_dim': 128},
        'net_arch': [64, 64],
        'activation_fn': torch.nn.ReLU,
    }

    tensorboard_log = logdir
    
    if algorithm == 'SAC':
        from stable_baselines3 import SAC
        
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            buffer_size=100000,
            learning_starts=10000,
            batch_size=256,
            tau=0.005,
            gamma=0.95,
            train_freq=4,
            gradient_steps=1,
            ent_coef="auto_0.1",         
            target_entropy="auto",
            policy_kwargs=policy_kwargs,
            tensorboard_log = tensorboard_log,
            verbose=1,
            **kwargs
        )
    
    return model

def create_env(coords, demands, capacity, num_vehicles, seed=None, vehicle_cost=50.0, max_customers=100):

    from dvrp_env import VRP
    from stable_baselines3.common.monitor import Monitor
    
    max_vehicles = max(3, min(15, len(demands) // 2))
    
    env = VRP(
        coords=coords,
        demands=demands, 
        capacity=capacity,
        max_vehicles=max_vehicles,
        seed = np.random.randint(0, 10000)
    )
    
    env = Wrap(env)
    env = Monitor(env)
    
    return env


def train_model(instance, total_steps=200000, vehicle_cost=50.0, max_customers=100, testfile = ""):

    from generate import generate_instance
    
    coords, demands, capacity, vehichle_count = load_data(testfile)
    
    env = create_env(coords, demands, capacity, 25, 
                           vehicle_cost=vehicle_cost, max_customers=max_customers)
    
    logdir = f"./logs/vrp/" + instance
    
    model = create_model(logdir, 'SAC', env)
    stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=10, verbose=1)

    # Evaluation callback that triggers stop callback
    eval_callback = EvalCallback(env, eval_freq=2000, callback_after_eval=stop_callback, verbose=1)

    # Train with progress bar and convergence callback
    model.learn(total_timesteps=total_steps, callback=[ProgressBarCallback(), eval_callback])
    env.close()
    return model


def main():
    
    algorithm = "SAC"
    
    test_file = "../../dataset/files/ " + input("Test file [c1_25.txt]:").strip() or "c1_25.txt"
    
    instance = input("Instance type (R/C/RC) [R]: ").strip().upper() or "R"

    steps_input = input("Total training steps [200000]: ").strip()
    total_steps = int(steps_input) if steps_input else 200000
    
    cost_input = input("Vehicle cost [50]: ").strip()
    vehicle_cost = float(cost_input) if cost_input else 50.0
    
    customer = (input("Max customers [100]: ").strip())
    customers = int(customer) if customer else 100
        
    # Train model
    model = train_model(
        instance = instance,
        total_steps=total_steps,
        vehicle_cost=vehicle_cost,
        max_customers=customers,
        testfile = test_file
    )
    
    model_path = f"{customers}_{total_steps//1000}"
    model.save(model_path)


if __name__ == "__main__":
    main()