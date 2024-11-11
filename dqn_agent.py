import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from pathlib import Path


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (np.array(state), np.array(action), np.array(reward, dtype=np.float32),
                np.array(next_state), np.array(done, dtype=np.uint8))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=128):
        self.state_size = state_size
        self.action_size = action_size
        self.device = "cpu"  # Force CPU usage, you can change to gpu if you want
        self.save_dir = Path("saved_models")
        self.save_dir.mkdir(exist_ok=True)

        # Q-Networks
        self.policy_net = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_net = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Training parameters
        self.batch_size = 256
        self.gamma = 0.99
        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_decay = 0.99998 #you can make this bigger for slower learning.
        self.target_update = 20
        self.learning_rate = 0.0005

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(500000) #i made this huge because I want my AI to have an extensive number of previous games to learn from
        self.steps_done = 0

    def select_action(self, state, training=True):

        epsilon = self.eps_end + (self.eps_start - self.eps_end) * \
                  np.exp(-1. * self.steps_done / 50000)

        if training:
            self.steps_done += 1


        if training and random.random() < epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)

            # Add small random noise to prevent identical actions
            q_values += torch.randn_like(q_values) * 0.1
            return q_values.max(1)[1].item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Compute Q(s_t, a)
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Compute max Q(s_{t+1}, a) for all next states
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            next_q_values[dones] = 0.0  # Set to 0 for terminal states

        # Compute expected Q values
        expected_q_values = rewards + (self.gamma * next_q_values)

        # Compute loss
        loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, episode):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }, self.save_dir / f"model_{episode}.pt")

    def load_model(self, path):
        try:
            #to check path and handle errors.
            if not Path(path).exists():
                print(f"\nError: Model file not found at {path}")
                print("Make sure you're using the correct path to your saved model.")
                print("\nExample paths:")
                print("- saved_models/model_800.pt")
                print("- ./saved_models/model_800.pt")
                return False

            checkpoint = torch.load(path, weights_only=True)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.steps_done = checkpoint['steps_done']
            print(f"\nSuccessfully loaded model from {path}")
            print(f"Training steps completed: {self.steps_done}")
            return True
        except Exception as e:
            print(f"\nError loading model: {str(e)}")
            print("Make sure you're using a valid model file.")
            return False