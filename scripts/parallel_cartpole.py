import torch 
import numpy as np 
import time
from copy import deepcopy
from envs import create_continuous_cartpole_env
from algorithms.ppo import Agent
from utils.logger import info, success, step, WebLogger
from gym.vector import AsyncVectorEnv
import multiprocessing

multiprocessing.set_start_method("fork", force=True)

HYPERPARAMETERS = {
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

# first thing is initialize all the environments
# run random actions on all of them
# graph how fast samples get collected over time over training

n_env = 5
num_episodes = 100

def make_env():
    return create_continuous_cartpole_env()

env = create_continuous_cartpole_env()
vec_env = AsyncVectorEnv([make_env for _ in range(n_env)])
obs_s = vec_env.reset()

input_dims = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = Agent(input_dims, n_actions, device, HYPERPARAMETERS)

for episode in range(num_episodes):
    total_reward = 0
    dones = [False] * n_env
    obs_s = vec_env.reset()
    n_dones = 0

    while n_dones < n_env:
        actions, log_probs, values = agent.choose_action(obs_s[0])
        actions = np.clip(actions, env.action_space.low, env.action_space.high)
        obs_next, rewards, dones, truncated, infos = vec_env.step(actions)

        if True in dones:
            n_dones += 1

        dw_flag = infos.get('TimeLimit.truncated', False)
        dw_flags = [dw_flag] * n_env
        agent.buffer.add(obs_s[0], actions, rewards, log_probs, values, obs_next, dones, dw_flags)
    
        obs_s = (obs_next, {})

    if False not in dones or agent.buffer.size() >= agent.batch_size:

        policy_loss, value_loss, entropy = agent.train()
        print("trained")

vec_env.close()