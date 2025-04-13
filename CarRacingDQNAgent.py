# CarRacingDQNAgent.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------- Prioritized Replay Buffer -----------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def add(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            self.priorities[self.pos] = max_priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return None, None, None
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # 정규화
        return indices, samples, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + 1e-5

# ----------------- Uniform Replay Buffer (PER 미사용 시) -----------------
class UniformReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=None):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        weights = torch.ones(batch_size, dtype=torch.float32)
        return indices, samples, weights

    def update_priorities(self, indices, td_errors):
        # URB에서는 업데이트할 우선순위가 없음.
        pass

# ----------------- Q-Network -----------------
class QNetwork(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=7, stride=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=4)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # 계산: 96x96 → conv1: ((96-7)//3+1)=30 → pool1: 30//2=15;
        # conv2: (15-4+1)=12 → pool2: 12//2=6 → 총 feature dim = 12*6*6 = 432.
        fc_input_size = 12 * 6 * 6
        self.fc1 = nn.Linear(fc_input_size, 216)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(216, num_actions)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.reshape(x.size(0), -1)  # view 대신 reshape() 사용
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# ----------------- DQN Agent -----------------
class CarRacingDQNAgent:
    def __init__(
        self,
        action_space = [
            (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2),
            (-1, 1, 0),   (0, 1, 0),   (1, 1, 0),
            (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),
            (-1, 0, 0),   (0, 0, 0),   (1, 0, 0)
        ],
        frame_stack_num = 3,
        memory_size = 5000,
        gamma = 0.95,
        epsilon = 1.0,
        epsilon_min = 0.1,
        epsilon_decay = 0.9999,
        learning_rate = 0.001,
        batch_size = 64,
        use_per = True,
        device = None
    ):
        self.action_space = action_space
        self.frame_stack_num = frame_stack_num
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.use_per = use_per
        
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNetwork(frame_stack_num, len(action_space)).to(self.device)
        self.target_model = QNetwork(frame_stack_num, len(action_space)).to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(memory_size)
        else:
            self.memory = UniformReplayBuffer(memory_size)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def memorize(self, state, action, reward, next_state, done):
        action_index = self.action_space.index(action)
        self.memory.add(state, action_index, reward, next_state, done)

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(self.action_space)
        state_tensor = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        action_index = torch.argmax(q_values, dim=1).item()
        return self.action_space[action_index]

    def replay(self, beta=0.4):
        if len(self.memory.buffer) < self.batch_size:
            return
        
        indices, minibatch, weights = self.memory.sample(self.batch_size, beta)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.tensor(np.array(states), dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        q_values = self.model(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            max_next_q_value = next_q_values.max(1)[0]
        target = rewards + self.gamma * max_next_q_value * (1 - dones)
        
        td_errors = (q_value - target).detach().cpu().numpy()
        weights = weights.to(self.device)
        loss = (weights * (q_value - target).pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory.update_priorities(indices, td_errors)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)

    def save(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
