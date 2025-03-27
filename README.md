# TarzanZero

TarzanZero is a reinforcement learning framework that implements various algorithms for training agents in environments.

## Features

- Proximal Policy Optimization (PPO) implementation
- CartPole continuous environment
- Interactive web dashboard for monitoring training
- Hyperparameter optimization using Optuna

## Prerequisites

Ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

## Running the CartPole Example

To train a PPO agent on the continuous CartPole environment:

```bash
python -m scripts.cartpole
```

This will start training with the default hyperparameters and open a web dashboard for monitoring.

## Hyperparameter Search

To find optimal hyperparameters for the PPO agent:

```bash
python -m scripts.hyperparam_search
```

This will start an Optuna-based hyperparameter search and show the progress on the web dashboard.

### Visualizing Search Results

After completing the hyperparameter search, you can visualize the results:

```bash
python -m scripts.visualize_search
```

This will generate plots showing:
- Reward history over trials
- Parameter importance analysis
- Summary of the best hyperparameters

Results are saved in the `results/` directory.

## Dashboard Features

The web dashboard allows you to:
- Monitor training progress in real time
- Pause and resume training
- Restart training from scratch
- Visualize rewards and losses
- Toggle different metrics on charts
- View hyperparameters

## Project Structure

- `algorithms/`: Implementation of reinforcement learning algorithms
- `buffers/`: Experience replay buffer implementations
- `config/`: Configuration and hyperparameter settings
- `envs/`: Environment implementations and wrappers
- `models/`: Neural network model definitions
- `scripts/`: Training and utility scripts
- `utils/`: Utility functions and logging tools
- `results/`: Saved results from hyperparameter searches

## Customizing Hyperparameters

You can customize the hyperparameter search range in `scripts/hyperparam_search.py` by modifying the search space in the `objective` function. 