from envpong import PongLogic
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return self.net(x)

class BotLeft:
    def __init__(self, env, is_training=False):
        self.env = env
        self.is_training = is_training
        
        self.input_shape = len(env.observation_space.sample())
        self.n_actions = 3 # UP, STILL, DOWN
        
        self.model = DQN(self.input_shape, self.n_actions)
        self.model_path = 'modelo_pong.pth'

        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
        
        # Parâmetros de treinamento
        if self.is_training:
            self.replay_memory = deque(maxlen=10000)
            self.learning_rate = 1e-4
            self.gamma = 0.99 
            self.epsilon_start = 1.0
            self.epsilon_end = 0.01
            self.epsilon_decay = 10000
            self.batch_size = 32
            self.steps_done = 0
            
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.loss_fn = nn.MSELoss()
            self.model.train() # Coloca o modelo em modo de treinamento
        else:
            self.model.eval() # Coloca o modelo em modo de avaliação

    def act(self, state):
        if self.is_training:
            self.steps_done += 1
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      torch.exp(torch.tensor(-1. * self.steps_done / self.epsilon_decay))
            
            if random.random() < epsilon:
                return random.choice([
                    PongLogic.PaddleMove.UP,
                    PongLogic.PaddleMove.STILL,
                    PongLogic.PaddleMove.DOWN
                ])

        with torch.no_grad():
            q_values = self.model(state)
            action_idx = torch.argmax(q_values).item()
            if action_idx == 0:
                return PongLogic.PaddleMove.DOWN
            elif action_idx == 1:
                return PongLogic.PaddleMove.STILL
            else: # action_idx == 2
                return PongLogic.PaddleMove.UP

    def observe(self, experience):
        if self.is_training:
            self.replay_memory.append(experience)

    def learn(self):
        if not self.is_training or len(self.replay_memory) < self.batch_size:
            return
        batch = random.sample(self.replay_memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(states, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.long)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        next_states_t = torch.tensor(next_states, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.bool)
        
        action_indices = actions_t + 1

        current_q_values = self.model(states_t).gather(1, action_indices.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_q_values = self.model(next_states_t).max(1)[0]
            next_q_values[dones_t] = 0.0

        target_q_values = rewards_t + self.gamma * next_q_values

        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class BotRight:
    def __init__(self, env):
        self.env = env
        self.obs = None
    
    def act(self):
        return random.choice([PongLogic.PaddleMove.DOWN, PongLogic.PaddleMove.STILL, PongLogic.PaddleMove.UP])  
    
    def observe(self, obs):
        self.obs = obs