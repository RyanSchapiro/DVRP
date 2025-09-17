Dynamic Vehicle Routing Problem with Deep Reinforcement Learning
This repository implements multiple approaches to solve the Dynamic Vehicle Routing Problem (DVRP), including classical heuristics and a novel Graph Attention Network (GAT) based reinforcement learning approach.
Overview
The project addresses the Dynamic Vehicle Routing Problem where:

Customers arrive dynamically over time according to a Poisson process
Only a fraction of customers are known initially (static customers)
Vehicles have limited capacity constraints
The objective is to minimize total travel distance plus vehicle fixed costs

Implemented Algorithms

Adaptive Large Neighborhood Search (ALNS) - Classical metaheuristic with destroy/repair operators
Clarke-Wright Savings Algorithm - Classical constructive heuristic adapted for dynamic arrivals
Deep Reinforcement Learning with GAT - Novel approach using Graph Attention Networks with Soft Actor-Critic (SAC)

Project Structure
├── alns_dvrp.py              # ALNS implementation for DVRP
├── clarke_wright_dvrp.py     # Clarke-Wright savings algorithm  
├── dvrp_env.py              # Gymnasium environment for DVRP
├── td.py                    # GAT-based RL training script
├── evaluation.py            # Model evaluation and benchmarking
├── util.py                  # Utility functions and data loading
├── data.py                  # Data generation and Poisson process
├── dataset/                 # Problem instances
│   └── files/              # VRP instance files (.txt format)
├── logs/                   # Training logs and tensorboard data
└── models/                 # Saved trained models
Dependencies
Core Requirements
bash# Deep Learning & RL
torch>=1.13.0
stable-baselines3>=1.8.0
gymnasium>=0.26.0
tensorboard>=2.10.0

# Scientific Computing
numpy>=1.21.0
pandas>=1.5.0
matplotlib>=3.5.0

# ALNS Implementation
alns>=5.0.0

# Additional
scipy>=1.9.0
tqdm>=4.64.0
Installation
bash# Create conda environment
conda create -n dvrp python=3.9
conda activate dvrp

# Install PyTorch (adjust for your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install stable-baselines3[extra] gymnasium alns pandas matplotlib tensorboard scipy tqdm
Dataset Format
VRP instances should be in standard VRPLIB format (.txt files):
NAME: instance_name
COMMENT: description
TYPE: CVRP
DIMENSION: n+1  # customers + depot
EDGE_WEIGHT_TYPE: EUC_2D
CAPACITY: vehicle_capacity

NODE_COORD_SECTION
0 x0 y0    # depot coordinates
1 x1 y1    # customer 1 coordinates
...

DEMAND_SECTION  
0 0        # depot has 0 demand
1 d1       # customer 1 demand
...
Place instance files in dataset/files/ directory.
Usage
1. ALNS for Dynamic VRP
bashpython alns_dvrp.py
# Enter filename when prompted (e.g., c1_25.txt)
Key Parameters:

STATIC_RATIO: Fraction of customers known initially (0.7 = 70%)
LAMBDA_RATE: Poisson arrival rate for dynamic customers
REOPT_ITERS: ALNS iterations per reoptimization (3000)
DESTROY_RATE: Fraction of customers to remove in destroy operators

2. Clarke-Wright Savings
bashpython clarke_wright_dvrp.py
# Enter filename when prompted
This implements the classical Clarke-Wright algorithm with complete reoptimization when new customers arrive.
3. Deep RL Training
bashpython td.py
Interactive Configuration:

Test file: Problem instance to train on
Instance type: R (random), C (clustered), RC (mixed)
Training steps: Default 200,000 timesteps
Vehicle cost: Fixed cost per vehicle (default 50)
Max customers: Maximum customers supported (default 100)

Training Features:

Graph Attention Network feature extraction
Soft Actor-Critic (SAC) algorithm
Custom action space wrapper for discrete VRP actions
Early stopping with evaluation callbacks
Tensorboard logging

4. Model Evaluation
bashpython evaluation.py
Evaluates trained RL models against benchmark instances:

Loads models automatically based on instance types
Runs multiple episodes per instance for statistical robustness
Compares against optimal solutions if available
Generates comprehensive performance metrics

Output Files:

RL_detailed.csv: Episode-by-episode results
RL_metrics.csv: Aggregated performance metrics
RL_best.json: Best solutions found per instance

Algorithm Details
ALNS (Adaptive Large Neighborhood Search)

Destroy Operators: Random removal, string-based removal
Repair Operators: Greedy insertion with capacity checking
Selection: Roulette wheel operator selection
Acceptance: Record-to-record travel acceptance criterion

RL with Graph Attention Networks

State Representation: Customer locations, demands, revealed status, vehicle state
Action Space: Continuous actions mapped to discrete customer selections
Reward: Negative travel distance (minimization objective)
Architecture: GAT feature extractor + SAC policy networks
Training: Multi-head attention for customer relationships

Dynamic Problem Handling
All algorithms handle dynamic arrivals by:

Starting with static customers only
Revealing new customers according to Poisson process
Reoptimizing solution when new customers arrive
Terminating when all customers served

Performance Metrics
The evaluation framework provides:
Solution Quality:

Best, average, worst costs across multiple runs
Optimality gap compared to known optimal solutions
Success and feasibility rates

Robustness:

Coefficient of variation for costs and computation times
Fraction of runs within 1% and 5% of optimal

Efficiency:

Average computation time per instance
Scalability with problem size

Configuration Parameters
Environment Parameters
pythonSTATIC_RATIO = 0.7      # 70% customers known initially  
LAMBDA_RATE = 0.4       # Poisson arrival rate
MAX_SIM_TIME = 1000     # Maximum simulation time steps
PENALTY = 1000          # Cost penalty for unassigned customers
RL Training Parameters
pythonlearning_rate = 1e-4    # SAC learning rate
buffer_size = 100000    # Replay buffer size  
batch_size = 256        # Mini-batch size
gamma = 0.95           # Discount factor
GAT Architecture
pythonembed_dim = 64         # Feature embedding dimension
num_heads = 4          # Multi-head attention heads
dropout = 0.1          # Dropout rate for regularization
Troubleshooting
Common Issues

CUDA Out of Memory

bash   # Reduce batch size in td.py
   batch_size = 128  # instead of 256

Instance File Format

Ensure files follow VRPLIB format exactly
Check that depot is node 0 with demand 0
Verify coordinate and demand sections are complete


Model Loading Errors

Ensure model files exist with .zip extension
Check model path mapping in get_path() function
Verify consistent max_customers parameter


Environment Issues

bash   # Reset environment if training fails
   conda deactivate
   conda activate dvrp
Performance Optimization

GPU Training: Ensure PyTorch detects CUDA for faster training
Parallel Evaluation: Increase num_runs in evaluation for better statistics
Memory Usage: Reduce buffer_size if running out of RAM

References

Pisinger, D. & Ropke, S. (2010). Large Neighborhood Search. Handbook of Metaheuristics.
Clarke, G. & Wright, J.W. (1964). Scheduling of Vehicles from a Central Depot to a Number of Delivery Points. Operations Research.
Veličković, P. et al. (2018). Graph Attention Networks. ICLR.
Haarnoja, T. et al. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning. ICML.

License
This project is released under the MIT License. See LICENSE file for details.
Contributing
Contributions are welcome! Please:

Fork the repository
Create a feature branch
Add tests for new functionality
Submit a pull request with clear description

For questions or issues, please open a GitHub issue with:

Problem description
System information
Steps to reproduce
Expected vs actual behavior

