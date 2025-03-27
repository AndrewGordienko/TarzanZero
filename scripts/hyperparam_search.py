import torch
import numpy as np
import optuna
import time
import json
import os
from datetime import datetime
from envs import create_continuous_cartpole_env
from algorithms.ppo import Agent
from utils.logger import info, success, step, warning, WebLogger

# Create directory for saving study results
os.makedirs('results', exist_ok=True)

# Initialize global variables
logger = None
best_params = {}
best_reward = -float('inf')
trial_number = 0
total_trials = 0
current_params = {}

def objective(trial):
    """Optuna objective function for PPO hyperparameter optimization"""
    global logger, best_params, best_reward, trial_number, total_trials, current_params
    
    # Increment trial counter
    trial_number += 1
    
    # Define hyperparameters to search
    actor_lr = trial.suggest_float("ACTOR_LR", 1e-5, 1e-3, log=True)
    critic_lr = trial.suggest_float("CRITIC_LR", 1e-5, 1e-3, log=True)
    entropy_coef_init = trial.suggest_float("ENTROPY_COEF_INIT", 0.01, 0.1)
    entropy_coef_decay = trial.suggest_float("ENTROPY_COEF_DECAY", 0.99, 0.999)
    gamma = trial.suggest_float("GAMMA", 0.98, 0.999)
    lambda_gae = trial.suggest_float("LAMBDA", 0.9, 0.99)
    kl_div_threshold = trial.suggest_float("KL_DIV_THRESHOLD", 0.001, 0.01)
    batch_size = trial.suggest_categorical("BATCH_SIZE", [256, 512, 1024])
    clip_ratio = trial.suggest_float("CLIP_RATIO", 0.1, 0.3)
    entropy_coef = trial.suggest_float("ENTROPY_COEF", 0.001, 0.01)
    value_loss_coef = trial.suggest_float("VALUE_LOSS_COEF", 0.5, 1.0)
    update_epochs = trial.suggest_int("UPDATE_EPOCHS", 4, 10)
    max_grad_norm = trial.suggest_float("MAX_GRAD_NORM", 0.1, 0.5)
    
    # Collect parameters
    params = {
        "ACTOR_LR": actor_lr,
        "CRITIC_LR": critic_lr,
        "ENTROPY_COEF_INIT": entropy_coef_init,
        "ENTROPY_COEF_DECAY": entropy_coef_decay,
        "GAMMA": gamma,
        "LAMBDA": lambda_gae,
        "KL_DIV_THRESHOLD": kl_div_threshold,
        "BATCH_SIZE": batch_size,
        "CLIP_RATIO": clip_ratio,
        "ENTROPY_COEF": entropy_coef,
        "VALUE_LOSS_COEF": value_loss_coef,
        "UPDATE_EPOCHS": update_epochs,
        "MAX_GRAD_NORM": max_grad_norm
    }
    
    # Update current parameters for display
    current_params = params
    
    # Set hyperparameters in logger for display
    search_info = {
        **params,
        "TRIAL": f"{trial_number}/{total_trials}",
        "BEST_REWARD_SO_FAR": best_reward
    }
    logger.set_hyperparameters(search_info)
    logger.update_status(f'Starting trial {trial_number}/{total_trials}')
    
    # Initialize the environment
    step(f"Initializing environment for trial {trial_number}/{total_trials}")
    env = create_continuous_cartpole_env()
    obs = env.reset()
    
    input_dims = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    
    # Initialize the agent with trial hyperparameters
    step("Initializing agent with trial hyperparameters")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(input_dims, n_actions, device, params)
    
    # Training parameters
    num_episodes = 500  # Shorter training for hyperparameter search
    max_steps_per_episode = 500
    trial_best_reward = -float('inf')
    trial_reward_sum = 0
    episodes_to_best_reward = 0  # Track when we reach the best reward
    
    # Reset episode-specific data in logger
    logger.reset_data()
    # Make sure restart flag is cleared at the beginning of each trial
    logger.clear_restart_flag()
    logger.set_hyperparameters(search_info)
    
    # Run training loop
    logger.update_status(f'Training trial {trial_number}/{total_trials}')
    step("Starting training loop")
    
    for episode in range(num_episodes):
        # Check if user requested a restart - if so, we'll skip to the next trial
        if logger.is_restart_requested():
            logger.clear_restart_flag()
            logger.update_status(f'Trial {trial_number}/{total_trials} restarted')
            info(f"Trial {trial_number}/{total_trials} restarted by user")
            env.close()
            return -float('inf')
            
        # Pause handling
        while logger.is_paused():
            logger.update_status(f'Trial {trial_number}/{total_trials} paused')
            time.sleep(0.5)
            
            # Check for restart while paused
            if logger.is_restart_requested():
                logger.clear_restart_flag()
                logger.update_status(f'Trial {trial_number}/{total_trials} restarted')
                info(f"Trial {trial_number}/{total_trials} restarted by user while paused")
                env.close()
                return -float('inf')
                
            continue
            
        # Resume status update
        if episode > 0 and logger.data['status'] == f'Trial {trial_number}/{total_trials} paused':
            logger.update_status(f'Training trial {trial_number}/{total_trials}')
        
        done = False
        obs = env.reset()
        total_reward = 0
        policy_loss = 0
        value_loss = 0
        entropy = 0
        
        step_count = 0
        while not done and step_count < max_steps_per_episode:
            step_count += 1
            # Check for pause or restart
            if logger.is_paused() or logger.is_restart_requested():
                break
                
            action, log_prob, value = agent.choose_action(obs)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            action = action[0]
            
            # Interact with the environment
            try:
                obs_next, reward, done, info_env = env.step(action)
                dw_flags = info_env.get('TimeLimit.truncated', False)
                
                agent.buffer.add(
                    obs,
                    action,
                    reward,
                    log_prob,
                    value,
                    obs_next,
                    done,
                    dw_flags
                )
                
                obs = obs_next
                total_reward += reward
                
                if done or agent.buffer.size() >= agent.batch_size:
                    policy_loss, value_loss, entropy = agent.train()
            except Exception as e:
                warning(f"Error in environment step: {e}")
                done = True
        
        # Skip this episode if restarted or paused
        if logger.is_restart_requested():
            logger.clear_restart_flag()
            logger.update_status(f'Trial {trial_number}/{total_trials} restarted')
            info(f"Trial {trial_number}/{total_trials} restarted by user")
            env.close()
            return -float('inf')
            
        if logger.is_paused():
            episode -= 1
            continue
        
        # Update best reward and track when we reach it
        if total_reward > trial_best_reward:
            trial_best_reward = total_reward
            episodes_to_best_reward = episode + 1  # +1 because episodes are 0-indexed
            info(f"New best reward in trial {trial_number}: {trial_best_reward:.2f} at episode {episodes_to_best_reward}")
        
        # Log episode results
        logger.log_episode(
            episode + 1,
            total_reward,
            trial_best_reward,
            policy_loss,
            value_loss,
            entropy
        )
        
        # Add to trial reward sum
        trial_reward_sum += total_reward
        
        # Update logger with progress
        if (episode + 1) % 10 == 0:
            info(f"Trial {trial_number}/{total_trials} - Episode {episode + 1}/{num_episodes} - " +
                 f"Avg Reward so far: {trial_reward_sum / (episode + 1):.2f}")
    
    # Trial completed
    trial_avg_reward = trial_reward_sum / num_episodes if num_episodes > 0 else -float('inf')
    
    # Update best parameters if this trial is better
    if trial_avg_reward > best_reward:
        best_reward = trial_avg_reward
        best_params = params
        
        # Save best parameters
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'results/best_params_{timestamp}.json', 'w') as f:
            json.dump({
                'params': best_params,
                'reward': best_reward,
                'trial': trial_number,
                'episodes_to_best': episodes_to_best_reward
            }, f, indent=4)
    
    # Update search info with results
    search_info["TRIAL_AVG_REWARD"] = trial_avg_reward
    search_info["BEST_REWARD_SO_FAR"] = best_reward
    search_info["EPISODES_TO_BEST"] = episodes_to_best_reward
    logger.set_hyperparameters(search_info)
    
    # Log trial results to the WebLogger for visualization
    logger.log_trial_result(
        trial_number, 
        trial_avg_reward, 
        trial_best_reward, 
        episodes_to_best_reward,  # Pass the episode count to best reward
        params
    )
    
    # Log trial results
    success(f"Trial {trial_number}/{total_trials} completed with average reward: {trial_avg_reward:.4f}")
    info(f"Best reward: {trial_best_reward:.4f} (reached at episode {episodes_to_best_reward})")
    info(f"Best average reward so far: {best_reward:.4f}")
    
    # Close environment
    env.close()
    
    return trial_avg_reward

def run_hyperparameter_search(n_trials=20):
    """Run the hyperparameter search with the specified number of trials"""
    global logger, best_params, best_reward, trial_number, total_trials, current_params
    
    # Initialize global variables
    trial_number = 0
    total_trials = n_trials
    best_reward = -float('inf')
    best_params = {}
    current_params = {}
    
    # Initialize the web logger with a different port (8080)
    logger = WebLogger(port=8080)
    logger.start()
    logger.update_status('Starting hyperparameter search')
    
    # Make sure any existing restart flag is cleared
    logger.clear_restart_flag()
    
    try:
        # Create an Optuna study
        study = optuna.create_study(direction="maximize")
        
        # Optimize with the given number of trials
        study.optimize(objective, n_trials=n_trials)
        
        # Log the best parameters
        logger.update_status('Hyperparameter search completed')
        best_trial = study.best_trial
        
        info("Best trial:")
        info(f"  Value: {best_trial.value:.4f}")
        info("  Params: ")
        for key, value in best_trial.params.items():
            info(f"    {key}: {value}")
        
        # Save the best parameters
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'results/final_best_params_{timestamp}.json', 'w') as f:
            json.dump({
                'params': best_trial.params,
                'reward': best_trial.value,
                'trial': study.best_trial.number
            }, f, indent=4)
        
        return best_trial.params
        
    except KeyboardInterrupt:
        warning("Search interrupted by user")
        
        # Save what we have so far
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'results/interrupted_best_params_{timestamp}.json', 'w') as f:
            json.dump({
                'params': best_params,
                'reward': best_reward,
                'trial': trial_number
            }, f, indent=4)
        
        return best_params

if __name__ == "__main__":
    # Number of trials for the search
    n_trials = 15  # Reduced number of trials for testing
    
    # Run the hyperparameter search
    best_hyperparams = run_hyperparameter_search(n_trials)
    
    # Print final best hyperparameters
    success("Hyperparameter search completed!")
    success("Best hyperparameters:")
    for param, value in best_hyperparams.items():
        success(f"  {param}: {value}") 