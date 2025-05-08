import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class DQNNet(nn.Module):
    def __init__(self, frame_stack_num, action_size):
        super(DQNNet, self).__init__()
        self.conv1 = nn.Conv2d(frame_stack_num, 6, kernel_size=7, stride=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=4)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        with torch.no_grad():
            dummy = torch.zeros(1, frame_stack_num, 96, 96)
            x = self.pool1(torch.relu(self.conv1(dummy)))
            x = self.pool2(torch.relu(self.conv2(x)))
            n_flatten = x.numel() // x.shape[0]
        self.fc1 = nn.Linear(n_flatten, 216)
        self.fc2 = nn.Linear(216, action_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CarRacingDQNAgent:
    def __init__(
        self,
        action_space = [
            (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2),
            (-1, 1,   0), (0, 1,   0), (1, 1,   0),
            (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),
            (-1, 0,   0), (0, 0,   0), (1, 0,   0)
        ],
        frame_stack_num = 3,
        memory_size = 5000,
        gamma = 0.95,
        epsilon = 1.0,
        epsilon_min = 0.1,
        epsilon_decay = 0.9999,
        learning_rate = 0.001,
        device = None
    ):
        self.action_space = action_space
        self.frame_stack_num = frame_stack_num
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = DQNNet(frame_stack_num, len(action_space)).to(self.device)
        self.target_model = DQNNet(frame_stack_num, len(action_space)).to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, eps=1e-7)
        self.loss_fn = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def memorize(self, state, action, reward, next_state, done):
        if isinstance(action, np.ndarray):
            action = tuple(action)
        self.memory.append((state, self.action_space.index(action), reward, next_state, done))

    def act(self, state):
        if np.random.rand() > self.epsilon:
            state_tensor = torch.FloatTensor(state).permute(2,0,1).unsqueeze(0).to(self.device)  # (H,W,C) -> (1,C,H,W)
            with torch.no_grad():
                act_values = self.model(state_tensor).cpu().numpy()[0]
            action_index = np.argmax(act_values)
        else:
            action_index = random.randrange(len(self.action_space))
        return self.action_space[action_index]

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = []
        targets = []
        for state, action_index, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).permute(2,0,1).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).permute(2,0,1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                target = self.model(state_tensor).cpu().numpy()[0]
            if done:
                target[action_index] = reward
            else:
                with torch.no_grad():
                    t = self.target_model(next_state_tensor).cpu().numpy()[0]
                target[action_index] = reward + self.gamma * np.amax(t)
            states.append(state)
            targets.append(target)
        states_tensor = torch.FloatTensor(np.array(states)).permute(0,3,1,2).to(self.device)
        targets_tensor = torch.FloatTensor(np.array(targets)).to(self.device)
        self.model.train()
        outputs = self.model(states_tensor)
        loss = self.loss_fn(outputs, targets_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name, map_location=self.device))
        self.update_target_model()

    def save(self, name):
        torch.save(self.target_model.state_dict(), name) 