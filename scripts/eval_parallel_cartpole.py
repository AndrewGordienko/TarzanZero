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
from rich.pretty import pprint
import optuna
import matplotlib.pyplot as plt

plt.ion()  # Interactive mode ON
fig, ax = plt.subplots(figsize=(7, 4))  # Use subplot for persistent access
plt.subplots_adjust(right=0.75)
multiprocessing.set_start_method("fork", force=True)
best_runs = {}  # {n_env: (avg_curve, max_curve)}
best_avg_reward = -np.inf
best_max_reward = -np.inf
best_actor_avg = None
best_actor_max = None

# actor_state_dict = torch.load("best_actor_avg.pth")
actor_state_dict = torch.load("best_actor_max.pth")

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

    # Run training and get both max and average reward
    max_reward, avg_reward, avg_curve, max_curve, agent = multi_trials(n_env=n_env, hyperparams=hyperparams)
    trial.set_user_attr("avg_curve", avg_curve)
    trial.set_user_attr("max_curve", max_curve)

    # Save metadata for inspection
    trial.set_user_attr("max_reward", max_reward)
    trial.set_user_attr("hyperparams", hyperparams)

    return avg_reward

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
    episode_rewards = []
    temp_avg = []
    temp_best = []

    input_dims = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(input_dims, n_actions, device, hyperparams)
    agent.actor.load_state_dict(actor_state_dict)
    
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
            episode_rewards.append(np.mean(env_rewards))
            avg_reward = np.mean(episode_rewards)
            temp_avg.append(avg_reward)
            temp_best.append(max_reward)

            if False not in dones or agent.buffer.size() >= agent.batch_size:
                start_time = time.time()
                # policy_loss, value_loss, entropy = agent.train()
                duration = time.time() - start_time
            
            row_data = [
                f"[white]{episode}[/white]",
                f"[green]{max_reward:.2f}[/green]",
                f"[blue]{avg_reward:.2f}[/blue]",
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
            table.add_column("Average Reward", justify="center")
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
    global best_avg_reward, best_max_reward, best_actor_avg, best_actor_max

    if avg_reward > best_avg_reward:
        best_avg_reward = avg_reward
        best_actor_avg = deepcopy(agent.actor.state_dict())

    if max_reward > best_max_reward:
        best_max_reward = max_reward
        best_actor_max = deepcopy(agent.actor.state_dict())

    return max_reward, avg_reward, temp_avg, temp_best, agent

trials = [1]
colors = ["red", "orange", "goldenrod", "green", "blue", "indigo", "violet"]

for n in trials:
    print(f"\n=== Running trials for n_env = {n} ===")
    study = optuna.create_study(direction="maximize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(lambda trial: objective(trial, n_env=n), n_trials=1)

    best_trial = study.best_trial
    avg_curve = best_trial.user_attrs["avg_curve"]
    max_curve = best_trial.user_attrs["max_curve"]
    best_runs[n] = (avg_curve, max_curve)

    print(f"\n[Best trial for n_env={n}]")
    print(f"  Episode Avg Reward: {best_trial.value:.2f}")
    print(f"  Max Reward: {best_trial.user_attrs['max_reward']:.2f}")
    print(f"  Hyperparameters:")
    pprint(best_trial.user_attrs["hyperparams"])

    avg_curve = best_trial.user_attrs["avg_curve"]
    max_curve = best_trial.user_attrs["max_curve"]
    best_runs[n] = (avg_curve, max_curve)

    # Plot to the existing axes without clearing
    color = colors[list(trials).index(n)]
    ax.plot(avg_curve, linestyle="--", color=color, label=f"n_env={n} Avg")
    ax.plot(max_curve, linestyle="-", color=color, label=f"n_env={n} Max")

    ax.set_title("Best Training Reward Curves per n_env")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)

plt.ioff()
plt.show()


