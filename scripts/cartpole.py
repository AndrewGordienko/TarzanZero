import torch
import numpy as np
import optuna
import time
from envs import create_continuous_cartpole_env
from algorithms.ppo import Agent
from utils.logger import info, success, step, WebLogger

# Fixed hyperparameters
AGENT = {
    "ACTOR_LR": 0.00021703891689103257,
    "CRITIC_LR": 0.0004271253752630734,
    "ENTROPY_COEF_INIT": 0.08801608152663996,
    "ENTROPY_COEF_DECAY": 0.9958697832658145,
    "GAMMA": 0.9908654911298033,
    "LAMBDA": 0.9580440174249956,
    "KL_DIV_THRESHOLD": 0.006302958474657382,
    "BATCH_SIZE": 512,
    "CLIP_RATIO": 0.27307659244433324,
    "ENTROPY_COEF": 0.003919224323254069,
    "VALUE_LOSS_COEF": 0.5820465065839804,
    "UPDATE_EPOCHS": 8,
    "MAX_GRAD_NORM": 0.33127351532987814
}

def run_training(logger, env, agent, num_episodes, max_steps_per_episode):
    """Run the training loop with support for pause and restart"""
    best_reward = -float('inf')
    total_reward_sum = 0
    
    logger.update_status('Training')
    step("Starting training loop")

    for episode in range(num_episodes):
        # Check if restart was requested
        if logger.is_restart_requested():
            logger.update_status('Restarting Training')
            # Clear the restart flag
            logger.clear_restart_flag()
            # Return so the training can be restarted
            return False
            
        # Check if training is paused
        while logger.is_paused():
            logger.update_status('Paused')
            time.sleep(0.5)  # Sleep to avoid busy waiting
            
            # Check for restart while paused
            if logger.is_restart_requested():
                logger.update_status('Restarting Training')
                logger.clear_restart_flag()
                return False
                
            continue
            
        # If we were paused and now resumed, update status
        if episode > 0 and logger.data['status'] == 'Paused':
            logger.update_status('Training')
            
        done = False
        obs = env.reset()
        total_reward = 0
        policy_loss = 0
        value_loss = 0
        entropy = 0

        while not done:
            # Check for pause or restart before each step
            if logger.is_paused() or logger.is_restart_requested():
                break
                
            action, log_prob, value = agent.choose_action(obs)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            action = action[0]
            # Interact with the environment
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
                
        # Check again for restart after episode
        if logger.is_restart_requested():
            logger.update_status('Restarting Training')
            logger.clear_restart_flag()
            return False
                
        # Skip the rest of this episode if paused
        if logger.is_paused():
            episode -= 1  # Retry this episode when resumed
            continue

        # Update best reward
        if total_reward > best_reward:
            best_reward = total_reward

        # Log episode results
        logger.log_episode(
            episode + 1,
            total_reward,
            best_reward,
            policy_loss,
            value_loss,
            entropy
        )

        # Track cumulative reward for evaluation
        total_reward_sum += total_reward

    logger.update_status('Training completed')
    return True

def objective():
    # Initialize the web logger
    logger = WebLogger(port=5000)
    logger.set_hyperparameters(AGENT)
    logger.start()
    
    while True:  # Main loop that allows for restarts
        # Update status
        logger.update_status('Initializing environment')
        
        # Initialize the environment
        step("Initializing the environment")
        env = create_continuous_cartpole_env()
        obs = env.reset()
        info(f"Initial observation: {obs}")

        input_dims = env.observation_space.shape[0]
        n_actions = env.action_space.shape[0]

        # Initialize the agent with fixed hyperparameters
        logger.update_status('Initializing agent')
        step("Initializing the agent")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent = Agent(input_dims, n_actions, device, AGENT)

        # Training parameters
        num_episodes = 5000
        max_steps_per_episode = 500
        
        # Run the training loop
        completed = run_training(logger, env, agent, num_episodes, max_steps_per_episode)
        
        env.close()
        
        if completed:
            success("Training completed successfully")
            break  # Exit the main loop if training completed normally
        else:
            success("Environment closed for restart")
            # Continue the loop to restart training
            time.sleep(1)  # Short delay before restarting

    # Return the final result
    return "Training complete"

if __name__ == "__main__":
    objective()
 
