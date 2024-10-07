import numpy as np
import random
from collections import namedtuple, deque

from learner.model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed = 0, 
                 replay_buffer_size = int(1e5), replay_batch_size = 64, update_every = 4,
                 target_gamma = 0.99, update_tau = 1e-3, lr = 5e-4, 
                 use_double_q = False, 
                 use_priorized_replay = False, prioritized_replay_eps = 0.001, prioritized_replay_alpha = 0.5):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.replay_batch_size = replay_batch_size
        self.replay_buffer_size = replay_buffer_size
        self.update_every = update_every
        self.target_gamma = target_gamma
        self.update_tau = update_tau
        self.lr = lr
        self.use_double_q = use_double_q
        self.use_priorized_replay = use_priorized_replay
        self.prioritized_replay_eps = prioritized_replay_eps
        self.prioritized_replay_alpha = prioritized_replay_alpha

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.replay_buffer_size, self.replay_batch_size, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        """Remember the state/action/reward/resulting_state tuple, and learn when applicable
        
        Params
        ======
            state (array_like): current state
            action (float): current action
            reward (float): reward obtained
            next_state (array_like): next state
            done (boolean): is terminal state
        """

        if self.use_priorized_replay:
            state = torch.from_numpy(state).float().unsqueeze(0)
            action = torch.from_numpy(np.array(action)).long().unsqueeze(0).unsqueeze(1)
            reward = torch.from_numpy(np.array(reward)).float().unsqueeze(0).unsqueeze(1)
            next_state = torch.from_numpy(next_state).float().unsqueeze(0)
            done = torch.from_numpy(np.array(done)).float().unsqueeze(0).unsqueeze(1)
            prob = np.sqrt(self.__calc_TD_err(state, action, reward, next_state, done, req_grad=False).squeeze().numpy()) + self.prioritized_replay_eps
            prob = np.power(prob, self.prioritized_replay_alpha)
        else:
            prob = float(1)

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done, prob)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.replay_batch_size:
                experiences = self.memory.sample(self.use_priorized_replay)
                self.__learn(experiences, self.target_gamma, self.use_double_q)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.int32(np.argmax(action_values.cpu().data.numpy()))
        else:
            return np.int32(random.choice(np.arange(self.action_size)))

    def __calc_TD_err(self, states, actions, rewards, next_states, dones, req_grad):
        """Calculate TD error as MSE
        """

        # Get max predicted Q values (for next states) from target model
        if self.use_double_q:
            # Use Double Q Network
            pred_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
            Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, pred_actions)
        else:
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (self.target_gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        if not req_grad:
            Q_expected = Q_expected.detach()

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        return loss

    def __learn(self, experiences, gamma, double_q):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Compute loss
        loss = self.__calc_TD_err(states, actions, rewards, next_states, dones, req_grad=True)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.__soft_update(self.qnetwork_local, self.qnetwork_target, self.update_tau)                     

    def __soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)
        self.probabilities = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        np.random.seed(seed)
        self.total_probability = float(0)
    
    def add(self, state, action, reward, next_state, done, probability=1):
        """Add a new experience to memory."""

        if len(self.probabilities) >= self.buffer_size:
            # If we're at the limit, subtract the probabilities of items to be removed from total probability 
            self.total_probability -= self.probabilities.popleft()

        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.total_probability += probability
        self.probabilities.append(probability)
    
    def sample(self, use_prioritized_replay):
        """Randomly sample a batch of experiences from memory."""
        if use_prioritized_replay:
            #experiences = random.choices(self.memory, weights=self.probabilities, k=self.batch_size)
            # We can't use random.choices because it can create duplicate entries
            picked = np.random.choice(len(self.memory), size=self.batch_size, replace=False, p=[p/self.total_probability for p in self.probabilities])
            experiences = [self.memory[i] for i in picked]
        else:
            experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)