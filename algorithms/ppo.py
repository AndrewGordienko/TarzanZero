import torch
import torch.optim as optim
import numpy as np
from models.actor_network import ActorNetwork
from models.critic_network import CriticNetwork
from buffers.replay_buffer import ReplayBuffer
from config.hyperparameters import AGENT
from rich.console import Console
import jax
import jax.numpy as jnp
console = Console()

class Agent:
    def __init__(self, input_shape, action_shape, device, agent_params):
        self.device = device
        self.agent_params = agent_params
        self.actor = ActorNetwork(input_shape, action_shape).to(self.device)
        self.critic = CriticNetwork(input_shape).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=agent_params["ACTOR_LR"])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=agent_params["CRITIC_LR"])
        self.buffer = ReplayBuffer()
        self.gamma = agent_params["GAMMA"]
        self.lambda_ = agent_params["LAMBDA"]
        self.clip_ratio = agent_params["CLIP_RATIO"]
        self.entropy_coef = agent_params["ENTROPY_COEF"]
        self.entropy_coef_decay = agent_params.get("ENTROPY_COEF_DECAY", 1.0)
        self.value_loss_coef = agent_params["VALUE_LOSS_COEF"]
        self.batch_size = agent_params["BATCH_SIZE"]
        self.update_epochs = agent_params["UPDATE_EPOCHS"]
        self.max_grad_norm = agent_params["MAX_GRAD_NORM"]
        self.input_shape = input_shape
        self.action_shape = action_shape

    def choose_action_jax(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        dist = self.actor.get_dist(state_tensor)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        value = self.critic(state_tensor)
        return (
            action.detach().cpu().numpy()[0],
            log_prob.detach().cpu().numpy()[0],
            value.detach().cpu().numpy()[0],
        )
    
    def choose_action(self, states):
        states_tensor = torch.FloatTensor(states).to(self.device)
        dist = self.actor.get_dist(states_tensor)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(-1)
        values = self.critic(states_tensor)

        return (
            actions.detach().cpu().numpy(),        # Shape: (num_envs, action_dim)
            log_probs.detach().cpu().numpy(),      # Shape: (num_envs,)
            values.detach().cpu().numpy(),         # Shape: (num_envs,)
        )

    def compute_gae(self, rewards, values, next_values, dones, dw_flags):
        # Flatten arrays to ensure they are 1D
        rewards = np.array(rewards).flatten()
        values = np.array(values).flatten()
        next_values = np.array(next_values).flatten()
        dones = np.array(dones).astype(int).flatten()
        dw_flags = np.array(dw_flags).astype(int).flatten()

        advantages = np.zeros_like(rewards)
        gae = 0
        for i in reversed(range(len(rewards))):
            mask = 0 if dones[i] and not dw_flags[i] else 1
            delta = rewards[i] + self.gamma * next_values[i] * mask - values[i]
            gae = delta + self.gamma * self.lambda_ * mask * gae
            advantages[i] = gae
        returns = advantages + values
        return advantages, returns
    
    def train(self):
        # Retrieve data from the buffer
        states, actions, rewards, log_probs, values, next_states, dones, dw_flags = self.buffer.get()

        # Flatten arrays to ensure they are 1D
        rewards = rewards.flatten()
        values = values.flatten()
        dones = dones.flatten()
        dw_flags = dw_flags.flatten()
        log_probs = log_probs.flatten()

        # Compute next values
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        next_values = self.critic(next_states_tensor).detach().cpu().numpy().flatten()

        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, next_values, dones, dw_flags)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # Ensure returns_tensor is 1D
        returns_tensor = returns_tensor.view(-1)

        # Dataloader for mini-batch updates
        dataset = torch.utils.data.TensorDataset(states_tensor, actions_tensor, old_log_probs_tensor,
                                                 advantages_tensor, returns_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.update_epochs):
            self.entropy_coef *= self.entropy_coef_decay
            for batch in loader:
                batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns = [
                    tensor.to(self.device) for tensor in batch
                ]

                # Value prediction
                values_pred = self.critic(batch_states).squeeze(-1)  # Shape: [batch_size]

                # Ensure batch_returns has the same shape as values_pred
                batch_returns = batch_returns.view(-1)

                # Value loss
                value_loss = torch.nn.functional.mse_loss(values_pred, batch_returns)

                # Get current policy distribution and log probabilities
                dist = self.actor.get_dist(batch_states)
                log_probs_tensor = dist.log_prob(batch_actions).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                # Calculate ratios
                ratios = torch.exp(log_probs_tensor - batch_old_log_probs)

                # Policy loss with PPO clipping
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Total loss
                total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                # Zero both optimizers
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                # Backward pass
                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                # Step optimizers
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        # Clear the buffer after update
        self.buffer.clear()

        # Return the losses for logging
        return policy_loss.item(), value_loss.item(), self.entropy_coef

