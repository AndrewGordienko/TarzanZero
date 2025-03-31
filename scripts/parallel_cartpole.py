import torch 
import numpy as np 
import time
from copy import deepcopy
from envs import create_continuous_cartpole_env
from algorithms.ppo import Agent
from utils.logger import info, success, step, WebLogger
from gym.vector import AsyncVectorEnv
import multiprocessing
import time
from rich.table import Table
from rich.live import Live
from rich import box
from rich.console import Console
import optuna

multiprocessing.set_start_method("fork", force=True)

def objective(trial, n_env):
    hyperparams = {
        "ACTOR_LR": trial.suggest_float("ACTOR_LR", 1e-5, 1e-3, log=True),
        "CRITIC_LR": trial.suggest_float("CRITIC_LR", 1e-5, 1e-3, log=True),
        "ENTROPY_COEF_INIT": trial.suggest_float("ENTROPY_COEF_INIT", 0.0, 0.1),
        "ENTROPY_COEF_DECAY": trial.suggest_float("ENTROPY_COEF_DECAY", 0.9, 1.0),
        "GAMMA": trial.suggest_float("GAMMA", 0.9, 0.999),
        "LAMBDA": trial.suggest_float("LAMBDA", 0.9, 1.0),
        "KL_DIV_THRESHOLD": trial.suggest_float("KL_DIV_THRESHOLD", 0.001, 0.01),
        "BATCH_SIZE": trial.suggest_categorical("BATCH_SIZE", [128, 256, 512, 1024]),
        "CLIP_RATIO": trial.suggest_float("CLIP_RATIO", 0.1, 0.4),
        "ENTROPY_COEF": trial.suggest_float("ENTROPY_COEF", 0.0, 0.01),
        "VALUE_LOSS_COEF": trial.suggest_float("VALUE_LOSS_COEF", 0.1, 1.0),
        "UPDATE_EPOCHS": trial.suggest_int("UPDATE_EPOCHS", 3, 10),
        "MAX_GRAD_NORM": trial.suggest_float("MAX_GRAD_NORM", 0.1, 1.0),
    }

    score = multi_trials(n_env=n_env, hyperparams=hyperparams)
    return score

def make_env():
    return create_continuous_cartpole_env()

def multi_trials(n_env, hyperparams):
    duration = None
    num_episodes = 100
    recent_rows = []
    scroll_limit = 10
    console = Console()
    env = create_continuous_cartpole_env()
    vec_env = AsyncVectorEnv([make_env for _ in range(n_env)])
    obs_s = vec_env.reset()
    samples_collected = 0
    max_reward = 0
    training_start = time.time()

    input_dims = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(input_dims, n_actions, device, hyperparams)
    
    with Live(console=console, refresh_per_second=4) as live:
        for episode in range(1, num_episodes):
            # Track each environment’s cumulative reward separately
            env_rewards = [0] * n_env
            dones = [False] * n_env
            obs_s = vec_env.reset()
            n_dones = 0

            while n_dones < n_env:
                actions, log_probs, values = agent.choose_action(obs_s[0])
                actions = np.clip(actions, env.action_space.low, env.action_space.high)
                obs_next, rewards, dones, truncated, infos = vec_env.step(actions)
                
                for i, r in enumerate(rewards):
                    env_rewards[i] += r

                if True in dones:
                    n_dones += 1

                dw_flag = infos.get('TimeLimit.truncated', False)
                dw_flags = [dw_flag] * n_env
                agent.buffer.add(obs_s[0], actions, rewards, log_probs, values, obs_next, dones, dw_flags)
                samples_collected += n_env
                obs_s = (obs_next, {})

            current_max = max(env_rewards)
            if current_max > max_reward:
                max_reward = current_max
                max_score_time = time.time() - training_start

            if False not in dones or agent.buffer.size() >= agent.batch_size:
                start_time = time.time()
                policy_loss, value_loss, entropy = agent.train()
                duration = time.time() - start_time
            
            row_data = [
                f"[white]{episode}[/white]",
                f"[green]{max_reward:.2f}[/green]",
                f"[white]{samples_collected / episode:.2f}[/white]",
                f"[white]{f'{duration:.2f}s' if duration is not None else 'N/A'}[/white]",
                f"[white]{max_score_time:.2f}s[/white]",
                f"[white]{n_env}[/white]",
            ]
            recent_rows.append(row_data)
            if len(recent_rows) > scroll_limit:
                recent_rows.pop(0)

            # Rebuild the table for display
            table = Table(
                title=f"Training Summary (up to episode {episode})",
                title_style="bold cyan",
                border_style="white",
                header_style="bold white",
                box=box.SIMPLE_HEAVY,
            )
            table.add_column("Episode", justify="center")
            table.add_column("High Score", justify="center")
            table.add_column("Avg Samples/Episode", justify="center")
            table.add_column("Duration", justify="center")
            table.add_column("Max Score Time", justify="center")
            table.add_column("n_env", justify="center")

            for row in recent_rows:
                table.add_row(*row)

            live.update(table)

            if max_reward >= 500:
                break

    vec_env.close()
    return max_reward

trials = [1, 5, 10, 50, 100, 500, 1000]

for n in trials:
    print(n)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, n_env=n), n_trials=5)

