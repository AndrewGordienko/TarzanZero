import numpy as np

class ReplayBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.next_states = []
        self.dones = []
        self.dws = []

    def add(self, states, actions, rewards, log_probs, values, next_states, done_flags, dw_flags):
        # Ensure scalar values are wrapped to be at least 1D arrays
        self.states.append(np.array(states))                     # Shape: (state_dim,)
        self.actions.append(np.array(actions))                   # Shape: (action_dim,)
        self.rewards.append(np.array([rewards]))                 # Shape: (1,)
        self.log_probs.append(np.array([log_probs]))             # Shape: (1,)
        self.values.append(np.array([values]))                   # Shape: (1,)
        self.next_states.append(np.array(next_states))           # Shape: (state_dim,)
        self.dones.append(np.array([done_flags]))                # Shape: (1,)
        self.dws.append(np.array([dw_flags]))                    # Shape: (1,)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.next_states = []
        self.dones = []
        self.dws = []

    def get(self):
        # Stack stored experiences into batches
        s = np.vstack(self.states)           # Shape: (batch_size, state_dim)
        a = np.vstack(self.actions)          # Shape: (batch_size, action_dim)
        r = np.concatenate(self.rewards, axis=0)        # Shape: (batch_size,)
        logprob_a = np.concatenate(self.log_probs, axis=0)  # Shape: (batch_size,)
        val = np.concatenate(self.values, axis=0)       # Shape: (batch_size,)
        s_next = np.vstack(self.next_states)            # Shape: (batch_size, state_dim)
        done = np.concatenate(self.dones, axis=0)       # Shape: (batch_size,)
        dw = np.concatenate(self.dws, axis=0)           # Shape: (batch_size,)
        return s, a, r, logprob_a, val, s_next, done, dw

    def size(self):
        return len(self.states)
