# SCVRP-RL

Implementation of differing methods for the stochastic capacitated vehicle routing problem (SCVRP). This repository combines classical heuristics, adaptive large neighbourhood search, reinforcement learning with Stable-Baselines3, and PyVRPs Genetic Algorithm solver as a baseline for adapted Solomon instances using a Poisson arrival model.

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Layout](#repository-layout)
- [Dataset Pipeline](#dataset-pipeline)
- [Environment Setup](#environment-setup)
- [Core Components](#core-components)
  - [Clarke-Wright Heuristic (`New/Ryan/CW`)](#clarke-wright-heuristic-newryancw)
  - [Adaptive Large Neighbourhood Search (`New/Ryan/ALNS`)](#adaptive-large-neighbourhood-search-newryanalns)
  - [SAC Reinforcement Learning Agent (`New/Ryan/RL/SB3`)](#sac-reinforcement-learning-agent-newryanrlsb3)
  - [PyVRP Genetic Baseline (`New/Ryan/solver.py`)](#pyvrp-genetic-baseline-newryansolverpy)
- [Evaluation Artifacts](#evaluation-artifacts)
- [Reproducing Experiments](#reproducing-experiments)
- [Results Snapshot](#results-snapshot)
- [Acknowledgements](#acknowledgements)

## Project Overview
- **Problem setting:** Customers are revealed through a Poisson process and must be serviced by capacitated vehicles. Each solver shares the same demand generator, distance computation, and vehicle penalty scheme (fixed cost of 50 per vehicle).
- **Algorithms implemented:**
  - Clarke-Wright savings heuristic adapted for dynamic arrivals.
  - Adaptive Large Neighbourhood Search with destroy/repair operators and Record-to-Record acceptance.
  - Soft Actor-Critic (SAC) agent with a Graph Attention Network feature extractor and action wrapper for discrete routing decisions.
- **Baseline:** PyVRP's genetic algorithm (GA) produces benchmark solutions that all other algorithms are evaluated against.
- **Outputs:** Detailed CSV metrics, JSON best-route summaries, PNG visualisations, and TensorBoard logs.

## Repository Layout
```
New/
├── Dataset/                    # Solomon instances and reducer utility
├── Ryan/
│   ├── CW/                     # Clarke-Wright solver + metrics pipeline
│   ├── ALNS/                   # ALNS implementation, utilities, benchmarking scripts
│   ├── RL/SB3/                 # SAC environment, training, and evaluation code
│   ├── solver.py               # PyVRP GA baseline driver
│   ├── plots.py                # Matplotlib comparison charts
│   ├── visualizer.py           # Instance and route plotting helper
│   ├── *_metrics.csv           # Aggregated statistics per algorithm
│   ├── *_detailed.csv          # Per-run logs
│   └── DVRP_SCHRYA010.pdf      # Project paper (latest manuscript)
├── optimal_solutions.json      # Reference best-known routes and costs
├── requirements.txt            # Python dependencies
└── README.md                   # You are here
```

## Dataset Pipeline
- Source files: `New/Dataset/solomon100_instances/*.txt` (original Solomon 100-customer instances).
- Reducer: `New/Dataset/reducer.py` trims each Solomon instance to 25, 50, and 100-customer variants while keeping the depot as index 0. Run it to regenerate reduced instances:
  ```bash
  python3 New/Dataset/reducer.py
  ```
- Working set: `New/Dataset/files/` contains the reduced instances consumed by every solver and evaluation script.
- Optimal references: `optimal_solutions.json` stores objective value, distance, and vehicle count for each instance, enabling gap analysis.

## Environment Setup
1. Create and activate a virtual environment (Python 3.10 or newer).
2. Install dependencies from `requirements.txt`:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Optional extras:
   - `tensorboard` for log inspection (already listed in requirements).
   - `tqdm` or `rich` if you want additional progress reporting.
   - GPU-enabled PyTorch build if training larger SAC models.

**Key packages explained:**
- `alns` powers the Adaptive Large Neighbourhood Search operators.
- `gymnasium`, `stable-baselines3`, `torch`, and `tensorboard` support the RL stack.
- `pyvrp` underpins the experimental solver in `New/Ryan/solver.py
- `matplotlib`, `seaborn`, `pandas`, and `numpy` back visualisation and data analysis scripts.

## Core Components

### Clarke-Wright Heuristic (`New/Ryan/CW`)
- `CW.py`: Implements a dynamic Clarke-Wright savings heuristic. Customers are partitioned into static and dynamic sets with `Poisson()` arrivals. The solver penalises unused customers and vehicle count, printing routes at termination.
  ```bash
  python3 New/Ryan/CW/CW.py
  ```
  You will be prompted for an instance file (e.g., `c1_50.txt`).
- `util.py`: Shared utilities for loading instances and generating Poisson arrivals (mirrors the RL data loader to maintain consistency).
- `metrics.py`: Automates multi-seed sweeps across the dataset, writing `clarke_wright_metrics.csv`, `clarke_wright_detailed_results.csv`, and `clarke_wright_best_routes.json`.
  ```bash
  python3 New/Ryan/CW/metrics.py
  ```

### Adaptive Large Neighbourhood Search (`New/Ryan/ALNS`)
- `ALNS.py`: Core ALNS loop with destroy (random/string) and repair (greedy insert) operators. It re-optimises whenever new customers arrive and terminates once the dynamic horizon is satisfied.
  ```bash
  python3 New/Ryan/ALNS/ALNS.py
  ```
  Enter a filename from `New/Dataset/files/` when prompted.
- `util.py`: ALNS-specific data loader and Poisson generator (identical interfaces to the CW utilities).
- `metrics.py`: Wraps `ALNS.py` in a batch evaluation harness, collects statistics over 20 seeds per instance, serialises `ALNS_metrics.csv`, `ALNS_detailed.csv`, and `ALNS_best.json`, and has demonstrated 100% feasibility across the evaluated Solomon subset.
  ```bash
  python3 New/Ryan/ALNS/metrics.py
  ```

### SAC Reinforcement Learning Agent (`New/Ryan/RL/SB3`)
- `dvrp_env.py`: Gymnasium environment with fixed-width observations (`[global_state, customer_slots]`) and action masking. Supports dynamic customer revelation, vehicle capacity tracking, and distance-based rewards.
- `td.py`: Training module defining the Graph Attention Network (`GATExtractor`), SAC policy instantiation, action wrapper (`Wrap`) that converts continuous SAC outputs into discrete routing decisions, and the training loop with Stable-Baselines3 callbacks.
  ```bash
  cd New/Ryan/RL/SB3
  python3 td.py
  ```
  Prompts request the problem file, instance family, training horizon, vehicle cost, and max customer count. Models are saved as `<customers>_<steps/1000>.zip`.
- `test.py`: Evaluation entry point that loads pre-trained SAC models (`c100_200.zip`, `r100_200.zip`, `rc100_200.zip`), runs multi-seed beam-search style rollouts via `multi_predict` and `beam_predict`, and records metrics in `RL_metrics.csv`, `RL_detailed.csv`, and `RL_best.json`.
  ```bash
  cd New/Ryan/RL/SB3
  python3 test.py
  ```
- `metrics.py`: Convenience wrapper for programme-driven RL evaluation, mirroring the CSV/JSON outputs of `test.py` for integration with the broader benchmarking pipeline.
- `data.py`, `generate.py`: Instance loader (matching the other solvers) and random instance generator for synthetic experiments.
- `logs/vrp/`: TensorBoard event files and helper plotting scripts (e.g., `curves.py` for quick Matplotlib summaries).

### PyVRP Genetic Baseline (`New/Ryan/solver.py`)
- `solver.py`: wraps [PyVRP](https://github.com/Pieter-JanVos/pyvrp)'s genetic algorithm search to generate the baseline solutions used for benchmarking. It constructs the instance via `pyvrp.Model`, sets the fixed vehicle cost to 50, registers the depot and customers, and lets the GA run for a configurable wall-clock budget (`MaxRuntime`).
- Baseline outputs (distance, total cost, vehicle count, and routes) should be synchronised with `optimal_solutions.json`, ensuring every heuristic/RL metric is computed as a gap to the GA-produced solution.
- Run the baseline for a specific instance:
  ```bash
  cd New/Ryan
  python3 solver.py
  ```
  
## Evaluation Artifacts
- `*_metrics.csv`: Aggregated statistics (success rate, feasibility rate, best/avg/worst cost, deviation from optimal, compute time, vehicle usage).
- `*_detailed.csv`: Per-seed records for deeper analysis or plotting.
- `*_best.json`: Best route per instance as discovered by the algorithm.
- PNG files (`success_rates.png`, `solution_costs.png`, `deviation_instance_type.png`, etc.) generated by `plots.py` summarise performance visually.
- Optimality gaps are calculated against the PyVRP GA baselines stored in `optimal_solutions.json`.

## Project Paper
- `New/Ryan/DVRP_SCHRYA010.pdf` contains the accompanying paper describing the experimental setup and empirical findings.

## Reproducing Experiments
1. Ensure datasets exist in `New/Dataset/files/` (regenerate with `reducer.py` if needed).
2. Install dependencies via `pip install -r requirements.txt`.
3. Run each evaluation pipeline:
   ```bash
   python3 New/Ryan/CW/metrics.py
   python3 New/Ryan/ALNS/metrics.py
   cd New/Ryan/RL/SB3 && python3 test.py
   cd ../../..    # return to repo root
   ```
4. Recommended: refresh the PyVRP GA baseline for any instances you plan to study:
   ```bash
   python3 New/Ryan/solver.py
   ```
5. Inspect results:
   - CSV summaries in `New/Ryan/`.
   - JSON route archives for visualisation or manual validation.
   - TensorBoard logs under `New/Ryan/RL/SB3/logs/vrp/`.
   - Static plots or custom dashboards generated with `plots.py`.

## Results Snapshot
Average results obtained across all instances (20 seeds per instance):

| Algorithm        | Success Rate | Avg Gap vs Baseline | Avg Solve Time |
|------------------|-------------:|--------------------:|---------------:|
| Clarke-Wright    | 1.00         | 4.26%               | 0.19 s         |
| ALNS             | 1.00         | 0.05%               | 27.71 s        |
| SAC (RL)         | 1.00         | 18.98%              | 0.50 s         |

- ALNS provides optimal/near-optimal solutions at the cost of runtime and maintained 100% feasibility across the tested instances.
- Clarke-Wright remains a fast, feasible baseline with modest gaps.
- The SAC agent achieves full completion but currently trails in cost.

## Acknowledgements
- Solomon benchmark suite for VRPTW instances.
- [ALNS Python package](https://github.com/ALNS-Python/alns) and [PyVRP](https://github.com/Pieter-JanVos/pyvrp) for ALNS backbone and Baseline solver.
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) and [Gymnasium](https://gymnasium.farama.org/) for reinforcement learning tooling.
- TensorBoard, Matplotlib, and Seaborn for visualizing and comparing results.

Questions, ideas, or contributions are always welcome—open an issue or start a discussion!
