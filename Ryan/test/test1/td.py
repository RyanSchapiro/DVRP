import torch
import torch.nn as nn
import numpy as np
from data import load_data
from dvrp_env import SCVRPEnv


# Import POMO components
class SCVRPPOMOExtractor(nn.Module):
    """Simplified POMO-style extractor for SB3 integration"""
    
    def __init__(self, observation_space, features_dim=128):
        super().__init__()
        self.features_dim = features_dim
        
        # Calculate observation structure
        obs_size = observation_space.shape[0]
        depot_features = 2  # x, y
        vehicle_features = 3  # current_node, load, time
        node_feature_size = 4  # x, y, demand, revealed
        
        # Calculate number of customers
        remaining = obs_size - depot_features - vehicle_features
        n_customers = remaining // node_feature_size
        
        # POMO-inspired architecture
        embedding_dim = 64
        
        # Embeddings
        self.depot_embed = nn.Linear(depot_features, embedding_dim)
        self.node_embed = nn.Linear(3, embedding_dim)  # x, y, demand
        self.vehicle_embed = nn.Linear(vehicle_features, embedding_dim)
        
        # Attention layers
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embedding_dim * 3, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim)
        )
        
        self.n_customers = n_customers
        self.depot_features = depot_features
        self.vehicle_features = vehicle_features
        self.node_feature_size = node_feature_size
    
    def forward(self, observations):
        batch_size = observations.shape[0]
        
        # Parse observations
        depot_coords = observations[:, :self.depot_features]
        
        node_start = self.depot_features
        node_end = node_start + self.n_customers * self.node_feature_size
        node_data = observations[:, node_start:node_end].view(batch_size, self.n_customers, self.node_feature_size)
        
        vehicle_state = observations[:, node_end:]
        
        # Extract node coordinates/demand and revealed status
        node_coords_demand = node_data[:, :, :3]  # x, y, demand
        node_revealed = node_data[:, :, 3:4]  # revealed status
        
        # Embeddings
        depot_emb = self.depot_embed(depot_coords).unsqueeze(1)  # (batch, 1, embed)
        node_emb = self.node_embed(node_coords_demand)  # (batch, n_customers, embed)
        vehicle_emb = self.vehicle_embed(vehicle_state)  # (batch, embed)
        
        # Combine depot and nodes
        graph_emb = torch.cat([depot_emb, node_emb], dim=1)  # (batch, n_nodes+1, embed)
        
        # Create attention mask for unrevealed customers
        mask = torch.zeros(batch_size, self.n_customers + 1, dtype=torch.bool, device=observations.device)
        mask[:, 1:] = (node_revealed.squeeze(-1) < 0.5)  # Mask unrevealed customers
        
        # Self-attention on graph
        graph_attended, _ = self.attention(graph_emb, graph_emb, graph_emb, key_padding_mask=mask)
        
        # Aggregate features
        depot_repr = graph_attended[:, 0]  # Depot representation
        node_repr = graph_attended[:, 1:].mean(dim=1)  # Average node representation
        
        # Combine all representations
        combined = torch.cat([depot_repr, node_repr, vehicle_emb], dim=1)
        output = self.output_proj(combined)
        
        return output


def create_env(coords, demands, capacity, num_vehicles, seed=None, vehicle_cost=50.0, max_customers=100):
    """Create SCVRP environment - fixed version"""
    return SCVRPEnv(
        coords=coords,
        demands=demands,
        capacity=capacity,
        vehicle_cost=vehicle_cost,
        seed=seed if seed is not None else np.random.randint(0, 10000)
    )


def create_model(logdir, algorithm='PPO', env=None, eval_env=None, **kwargs):
    """Create model with POMO-style features"""
    
    policy_kwargs = {
        'features_extractor_class': SCVRPPOMOExtractor,
        'features_extractor_kwargs': {'features_dim': 128},
        'net_arch': dict(pi=[256, 128], vf=[256, 128]),
        'activation_fn': torch.nn.ReLU,
    }

    if algorithm == 'PPO':
        try:
            from sb3_contrib import MaskablePPO
            from sb3_contrib.common.wrappers import ActionMasker
            from stable_baselines3.common.monitor import Monitor
            from stable_baselines3.common.callbacks import EvalCallback
            
            def mask_fn(env_instance):
                return env_instance.action_masks()
            
            # Wrap training environment
            env = ActionMasker(env, mask_fn)
            env = Monitor(env)
            
            model = MaskablePPO(
                "MlpPolicy",
                env,
                learning_rate=1e-4,
                n_steps=512,
                batch_size=128,
                n_epochs=4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=policy_kwargs,
                tensorboard_log=logdir,
                verbose=1,
                **kwargs
            )
            
            # Setup evaluation
            if eval_env is not None:
                eval_env = ActionMasker(eval_env, mask_fn)
                eval_env = Monitor(eval_env)
                
                eval_callback = EvalCallback(
                    eval_env,
                    best_model_save_path=f"{logdir}/best_model",
                    log_path=f"{logdir}/eval_logs",
                    eval_freq=10000,
                    deterministic=True,
                    render=False,
                    verbose=1
                )
                
                return model, eval_callback
                
        except ImportError:
            print("sb3-contrib not available, using regular PPO")
            from stable_baselines3 import PPO
            from stable_baselines3.common.monitor import Monitor
            
            env = Monitor(env)
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=1e-4,
                n_steps=512,
                batch_size=128,
                n_epochs=4,
                gamma=0.99,
                policy_kwargs=policy_kwargs,
                tensorboard_log=logdir,
                verbose=1,
                **kwargs
            )
            return model, None
    
    return None, None


def train_model(instance, total_steps=100000, vehicle_cost=50.0, max_customers=100, testfile=""):
    """Train POMO-style SCVRP model"""
    coords, demands, capacity, vehicle_count = load_data(testfile)
    
    print(f"Loaded problem: {len(demands)-1} customers, capacity: {capacity}")
    
    # Create environments
    train_env = create_env(coords, demands, capacity, vehicle_count, 
                          vehicle_cost=vehicle_cost, max_customers=max_customers)
    
    eval_env = create_env(coords, demands, capacity, vehicle_count, 
                         vehicle_cost=vehicle_cost, max_customers=max_customers, 
                         seed=999)
    
    logdir = f"./logs/scvrp_pomo/{instance}"
    
    # Create model
    model, eval_callback = create_model(logdir, 'PPO', train_env, eval_env=eval_env)
    
    if model is None:
        print("Failed to create model")
        return None
    
    print(f"Starting POMO-style SCVRP training for {total_steps} timesteps...")
    
    # Training
    if eval_callback is not None:
        model.learn(
            total_timesteps=total_steps,
            callback=eval_callback,
            progress_bar=True
        )
    else:
        model.learn(
            total_timesteps=total_steps,
            progress_bar=True
        )
    
    train_env.close()
    if eval_env:
        eval_env.close()
    
    return model


def main():
    print("SCVRP POMO-Style Training")
    print("=" * 40)
    
    test_file = "../../../dataset/files/" + (input("Test file [c1_25.txt]: ").strip() or "c1_25.txt")
    instance = input("Instance type (R/C/RC) [R]: ").strip().upper() or "R"
    
    steps_input = input("Total training steps [100000]: ").strip()
    total_steps = int(steps_input) if steps_input else 100000
    
    cost_input = input("Vehicle cost [50]: ").strip()
    vehicle_cost = float(cost_input) if cost_input else 50.0
    
    customer = input("Max customers [100]: ").strip()
    customers = int(customer) if customer else 100
    
    # Training
    model = train_model(
        instance=instance,
        total_steps=total_steps,
        vehicle_cost=vehicle_cost,
        max_customers=customers,
        testfile=test_file
    )
    
    if model is not None:
        model_path = f"scvrp_pomo_{instance.lower()}_{total_steps//1000}k"
        model.save(model_path)
        print(f"Model saved to {model_path}")
    else:
        print("Training failed")


if __name__ == "__main__":
    main()