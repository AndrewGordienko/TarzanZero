import os
import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
from datetime import datetime

def load_results(results_dir='results'):
    """Load all hyperparameter search result files"""
    # Find all result files
    result_files = glob.glob(os.path.join(results_dir, 'best_params_*.json'))
    
    if not result_files:
        print("No result files found in the results directory.")
        return None
    
    # Sort by timestamp (newest first)
    result_files.sort(reverse=True)
    
    # Load all results
    all_results = []
    for file_path in result_files:
        with open(file_path, 'r') as f:
            try:
                result = json.load(f)
                # Extract timestamp from filename
                timestamp_str = file_path.split('_')[-1].split('.')[0]
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
                result['timestamp'] = timestamp
                all_results.append(result)
            except json.JSONDecodeError:
                print(f"Error parsing {file_path}. Skipping.")
    
    return all_results

def plot_reward_history(results):
    """Plot the reward history over trials"""
    if not results:
        print("No results to plot.")
        return
    
    # Sort by trial number
    results.sort(key=lambda x: x.get('trial', 0))
    
    trials = [r.get('trial', i) for i, r in enumerate(results)]
    rewards = [r.get('reward', 0) for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(trials, rewards, 'o-', color='blue')
    plt.axhline(y=max(rewards), color='red', linestyle='--', label=f'Best: {max(rewards):.2f}')
    plt.title('Reward History Over Trials')
    plt.xlabel('Trial Number')
    plt.ylabel('Average Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the figure
    plt.savefig(os.path.join('results', 'reward_history.png'))
    plt.close()
    
    print(f"Reward history plot saved to results/reward_history.png")

def plot_parameter_importance(results):
    """Plot the parameter values for the top performing trials"""
    if not results:
        print("No results to plot.")
        return
    
    # Sort by reward (best first)
    results.sort(key=lambda x: x.get('reward', 0), reverse=True)
    
    # Take top 5 results
    top_results = results[:min(5, len(results))]
    
    # Get all parameter names
    all_params = []
    for result in top_results:
        if 'params' in result:
            all_params.extend(result['params'].keys())
    all_params = sorted(set(all_params))
    
    # Create figure with subplots
    n_params = len(all_params)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()
    
    # Plot each parameter
    for i, param in enumerate(all_params):
        if i < len(axes):
            ax = axes[i]
            
            # Get values for this parameter
            values = []
            rewards = []
            
            for result in results:
                if 'params' in result and param in result['params']:
                    values.append(result['params'][param])
                    rewards.append(result['reward'])
            
            # Plot parameter values vs rewards
            if values:
                ax.scatter(values, rewards, alpha=0.5)
                ax.set_title(param)
                ax.grid(True, alpha=0.3)
    
    # Hide any unused subplots
    for i in range(n_params, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'parameter_importance.png'))
    plt.close()
    
    print(f"Parameter importance plot saved to results/parameter_importance.png")

def create_summary_table(results):
    """Create a summary table of the best hyperparameters"""
    if not results:
        print("No results to create summary table.")
        return
    
    # Sort by reward (best first)
    results.sort(key=lambda x: x.get('reward', 0), reverse=True)
    
    # Take best result
    best_result = results[0]
    
    # Create a formatted table of parameters
    if 'params' in best_result:
        print("\nBest Hyperparameters:")
        print("-" * 50)
        print(f"Average Reward: {best_result.get('reward', 0):.4f}")
        print(f"Trial Number: {best_result.get('trial', 'N/A')}")
        print("-" * 50)
        print("Parameter             Value")
        print("-" * 50)
        
        for param, value in sorted(best_result['params'].items()):
            if isinstance(value, float):
                print(f"{param:<20} {value:.6f}")
            else:
                print(f"{param:<20} {value}")
        
        print("-" * 50)
        
        # Save to file
        with open(os.path.join('results', 'best_params_summary.txt'), 'w') as f:
            f.write(f"Best Hyperparameters:\n")
            f.write(f"Average Reward: {best_result.get('reward', 0):.4f}\n")
            f.write(f"Trial Number: {best_result.get('trial', 'N/A')}\n")
            f.write("-" * 50 + "\n")
            
            for param, value in sorted(best_result['params'].items()):
                if isinstance(value, float):
                    f.write(f"{param:<20} {value:.6f}\n")
                else:
                    f.write(f"{param:<20} {value}\n")
        
        print(f"Summary saved to results/best_params_summary.txt")

if __name__ == "__main__":
    # Load results
    results = load_results()
    
    if results:
        # Plot reward history
        plot_reward_history(results)
        
        # Plot parameter importance
        plot_parameter_importance(results)
        
        # Create summary table
        create_summary_table(results)
    else:
        print("No results found. Run a hyperparameter search first.") 